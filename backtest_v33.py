import argparse, math, time, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd, requests
from requests.adapters import HTTPAdapter

PUBLIC_BASE="https://data.binance.vision"
MARKET_DATA_HOST="https://data-api.binance.vision"
CACHE=Path("cache/binance_public"); CACHE.mkdir(parents=True,exist_ok=True)
SOURCE_STATS={"public_zip_success":0,"public_zip_missing":0,"market_api_success":0,"market_api_failed":0}
KLINE_COLS=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
MAX_RETRIES=5
BACKOFF_BASE=1.0
SYMBOL_THROTTLE_SECONDS=0.20

SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"crypto-scanner-v3.3-binance-public-data/1.0"})
SESSION.mount("https://",HTTPAdapter(pool_connections=10,pool_maxsize=10,max_retries=0))
SESSION.mount("http://",HTTPAdapter(pool_connections=10,pool_maxsize=10,max_retries=0))

def clip(x,a=0,b=100): return float(max(a,min(b,0 if pd.isna(x) else x)))
def tsms(x): return int(pd.Timestamp(x).timestamp()*1000)

def _parse_open_time(s):
    v=pd.to_numeric(s,errors="coerce")
    out=pd.Series(pd.NaT,index=v.index,dtype="datetime64[ns]")
    ms=v.notna() & (v.abs()<1e14)
    us=v.notna() & ~ms
    if ms.any(): out.loc[ms]=pd.to_datetime(v.loc[ms],unit="ms",errors="coerce")
    if us.any(): out.loc[us]=pd.to_datetime(v.loc[us],unit="us",errors="coerce")
    return out

def _normalize_frame(d):
    if d.empty:return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    for c in ["open","high","low","close","volume"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["date","open","high","low","close","volume"])
    return d[["date","open","high","low","close","volume"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)

def _parse_zip_bytes(content):
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names=[n for n in z.namelist() if not n.endswith("/")]
            if not names: raise RuntimeError("ZIP contains no CSV file")
            raw=z.read(names[0])
        d=pd.read_csv(io.BytesIO(raw),header=None,names=KLINE_COLS)
    except Exception as e:
        raise RuntimeError(f"PUBLIC_ZIP_PARSE_FAILED: {e}") from e
    if d.empty:return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    d["open_time"]=pd.to_numeric(d["open_time"],errors="coerce")
    d=d[d["open_time"].notna()].copy()
    d["date"]=_parse_open_time(d["open_time"])
    return _normalize_frame(d)

def _month_cache_paths(sym,period):
    p=CACHE/"monthly"/sym; p.mkdir(parents=True,exist_ok=True)
    stem=f"{sym}-1d-{period.year:04d}-{period.month:02d}"
    return p/f"{stem}.zip",p/f"{stem}.csv"

def _month_url(sym,period):
    name=f"{sym}-1d-{period.year:04d}-{period.month:02d}.zip"
    return f"{PUBLIC_BASE}/data/spot/monthly/klines/{sym}/1d/{name}"

def _sleep_backoff(attempt):
    time.sleep(BACKOFF_BASE*(2**(attempt-1)))

def _request_with_retry(url,*,params=None,timeout=30,sym=None,month=None,max_retries=MAX_RETRIES):
    last_error=None
    for attempt in range(1,max_retries+1):
        try:
            r=SESSION.get(url,params=params,timeout=timeout)
            if r.status_code==404:
                return r,attempt
            if 500<=r.status_code<=599:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}",response=r)
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r,attempt
        except (requests.exceptions.ConnectionError,requests.exceptions.Timeout,requests.exceptions.ChunkedEncodingError,requests.exceptions.HTTPError) as e:
            last_error=e
            label=str(month) if month is not None else "n/a"
            print(f"DOWNLOAD_RETRY symbol={sym or 'n/a'} month={label} retry_count={attempt} error={type(e).__name__}: {e}",flush=True)
            if attempt<max_retries:
                _sleep_backoff(attempt)
        except Exception:
            raise
    label=str(month) if month is not None else "n/a"
    print(f"DOWNLOAD_FAILED symbol={sym or 'n/a'} month={label} retry_count={max_retries} error={type(last_error).__name__}: {last_error}",flush=True)
    raise RuntimeError(f"DOWNLOAD_FAILED symbol={sym or 'n/a'} month={label} retry_count={max_retries} error={last_error}") from last_error

def fetch_public_month(sym,period,count_stats=True,use_cache=True):
    zip_path,csv_path=_month_cache_paths(sym,period); url=_month_url(sym,period)
    if use_cache and csv_path.exists():
        try:
            d=pd.read_csv(csv_path,parse_dates=["date"])
            d=_normalize_frame(d)
            if not d.empty:
                if count_stats: SOURCE_STATS["public_zip_success"]+=1
                return d
        except Exception:
            csv_path.unlink(missing_ok=True)
    if use_cache and zip_path.exists():
        try:
            d=_parse_zip_bytes(zip_path.read_bytes())
            if not d.empty:
                d.to_csv(csv_path,index=False)
                if count_stats: SOURCE_STATS["public_zip_success"]+=1
                return d
        except Exception:
            zip_path.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)
    r,attempts=_request_with_retry(url,timeout=30,sym=sym,month=period)
    if r.status_code==404:
        if count_stats: SOURCE_STATS["public_zip_missing"]+=1
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    d=_parse_zip_bytes(r.content)
    if d.empty:
        raise RuntimeError(f"PUBLIC_ZIP_EMPTY_AFTER_PARSE symbol={sym} month={period} retry_count={attempts}")
    zip_path.write_bytes(r.content)
    d.to_csv(csv_path,index=False)
    if count_stats: SOURCE_STATS["public_zip_success"]+=1
    return d

def fetch_market_api(sym,start,end):
    rows=[]; cur=pd.Timestamp(start)
    try:
        while cur<=end:
            r,_=_request_with_retry(
                MARKET_DATA_HOST+"/api/v3/klines",
                params=dict(symbol=sym,interval="1d",startTime=tsms(cur),endTime=tsms(end+pd.Timedelta(days=1))-1,limit=1000),
                timeout=20,sym=sym,month=f"market_api:{cur.date()}"
            )
            if r.status_code==404: raise RuntimeError("HTTP 404")
            b=r.json()
            if not b: break
            rows += [[x[0],x[1],x[2],x[3],x[4],x[5]] for x in b]
            nxt=pd.to_datetime(b[-1][0],unit="ms")+pd.Timedelta(days=1)
            if nxt<=cur: break
            cur=nxt
            time.sleep(.05)
        if not rows: raise RuntimeError("NO_DATA")
        d=pd.DataFrame(rows,columns=["open_time","open","high","low","close","volume"])
        d["date"]=pd.to_datetime(pd.to_numeric(d["open_time"],errors="coerce"),unit="ms",errors="coerce")
        d=_normalize_frame(d)
        d=d[(d.date>=start)&(d.date<=end)].copy()
        if d.empty: raise RuntimeError("NO_DATA_IN_RANGE")
        SOURCE_STATS["market_api_success"]+=1
        return d
    except Exception:
        SOURCE_STATS["market_api_failed"]+=1
        raise

def fetch(sym,start,end):
    months=pd.period_range(start=pd.Timestamp(start).to_period("M"),end=pd.Timestamp(end).to_period("M"),freq="M")
    parts=[]; missing=[]
    for period in months:
        d=fetch_public_month(sym,period)
        if d.empty: missing.append(period)
        else: parts.append(d)
    used_api=False
    now=pd.Timestamp.utcnow().tz_localize(None)
    recent_floor=(now-pd.Timedelta(days=62)).to_period("M")
    recent_missing=[p for p in missing if p>=recent_floor]
    if recent_missing:
        gap_start=max(pd.Timestamp(start),recent_missing[0].start_time)
        if parts:
            last=max(x.date.max() for x in parts)
            gap_start=max(gap_start,last+pd.Timedelta(days=1))
        if gap_start<=end:
            try:
                api=fetch_market_api(sym,gap_start,end)
                parts.append(api); used_api=True
            except Exception:
                pass
    if not parts:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"]),"binance_public_zip"
    d=pd.concat(parts,ignore_index=True)
    d=d.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    d=d[(d.date>=start)&(d.date<=end)].copy().reset_index(drop=True)
    source="binance_market_data_api" if used_api else "binance_public_zip"
    return d,source

def preflight_public_zip(start,end):
    months=list(pd.period_range(start=pd.Timestamp(start).to_period("M"),end=pd.Timestamp(end).to_period("M"),freq="M"))
    now_month=pd.Timestamp.utcnow().tz_localize(None).to_period("M")
    candidates=[p for p in months if p<now_month]
    if len(candidates)<2:
        raise RuntimeError("PRE-FLIGHT FAILED: fewer than two completed months exist in requested range")
    test_months=candidates[:2]
    symbols=["BTCUSDT","ETHUSDT","ZRXUSDT"]
    total=len(symbols)*len(test_months); passed=0; failures=[]
    print(f"PRE-FLIGHT TEST: symbols={symbols} months={[str(p) for p in test_months]} tests={total}")
    for sym in symbols:
        for period in test_months:
            try:
                d=fetch_public_month(sym,period,count_stats=False,use_cache=True)
                if d.empty:
                    raise RuntimeError("ZIP missing/404 or empty")
                if len(d)<20:
                    raise RuntimeError(f"parsed rows={len(d)} (<20)")
                passed+=1
                print(f"PRE-FLIGHT PASS symbol={sym} month={period} rows={len(d)}",flush=True)
            except Exception as e:
                failures.append(f"{sym} {period}: {e}")
                print(f"PRE-FLIGHT FAIL symbol={sym} month={period} error={e}",flush=True)
    rate=passed/total if total else 0
    print(f"PRE-FLIGHT SUCCESS RATE: {passed}/{total} = {rate:.1%}")
    if passed<total:
        raise RuntimeError("PRE-FLIGHT FAILED: "+" | ".join(failures[:6]))
    return rate

def indicators(d):
    d=d.copy(); pc=d.close.shift()
    tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d["atr14"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    delta=d.close.diff(); signed=np.where(delta>0,d.volume,np.where(delta<0,-d.volume,0))
    d["obv"]=pd.Series(signed,index=d.index).cumsum()
    den=(d.high-d.low).replace(0,np.nan); mfm=(((d.close-d.low)-(d.high-d.close))/den).fillna(0)
    d["cmf20"]=(mfm*d.volume).rolling(20).sum()/d.volume.rolling(20).sum()
    gain=delta.clip(lower=0); loss=(-delta.clip(upper=0))
    ag=gain.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); al=loss.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=ag/al.replace(0,np.nan); d["rsi14"]=100-100/(1+rs)
    for n in [20,60,120]: d[f"ma{n}"]=d.close.rolling(n).mean()
    d["vma20"]=d.volume.rolling(20).mean(); d["vr20"]=d.volume/d.vma20
    return d

def pscore(r):
    return 100 if r<=.2 else 85 if r<=.5 else 65 if r<=1 else 35 if r<=2 else 15 if r<=3 else 0
def epscore(r):
    return 100 if r<=.1 else 80 if r<=.3 else 60 if r<=.6 else 35 if r<=1 else 15 if r<=2 else 0
def mscore(m):
    return 20 if m<=0 else 60 if m<=.1 else 100 if m<=.3 else 70 if m<=.6 else 35 if m<=1 else 0
def cscore(x):
    return 100 if x<=.05 else 75 if x<=.1 else 50 if x<=.2 else 25 if x<=.3 else 0

def E(d,ci):
    w=d.loc[max(20,ci-59):ci]; ev=w[w.vr20>=2]
    if ev.empty:return dict(score=0,event=None,abs=0,storage=False,stall=False)
    ei=int(ev[ev.vr20==ev.vr20.max()].index[-1]); vr=float(d.loc[ei,"vr20"])
    ve=clip(((vr-1)/4)*100); pre=max(0,ei-5); c0=float(d.loc[pre,"close"]); ce=float(d.loc[ei,"close"])
    pn=epscore(ce/c0-1 if c0 else 99); atr=float(d.loc[ei,"atr14"]) if pd.notna(d.loc[ei,"atr14"]) else 0
    post=d.loc[ei:min(ci,ei+30)]; pl=float(post.low.min()); el=float(d.loc[ei,"low"])
    a=max(0,min(1,1-max(0,el-pl)/(2*atr))) if atr>0 else 0
    s=.4*ve+.3*pn+.3*(100*a)
    if atr>0 and pl<el-2*atr:s=min(s,59)
    pre20=d.loc[max(0,ei-20):ei-1]; post7=d.loc[ei+1:min(ci,ei+7)]
    storage=False
    if len(pre20) and len(post7) and ce!=c0:
        stor=(float(post7.close.iloc[-1])-c0)/(ce-c0)
        storage=stor>0 and float(post7.low.min())>=float(pre20.low.min()) and float(post7.close.median())>float(pre20.close.median())
    last60=d.loc[max(0,ci-59):ci]; prev20=d.loc[max(0,ci-39):max(0,ci-20)]; rec20=d.loc[max(0,ci-19):ci]
    stall=False
    if int((last60.vr20>=2).sum())>=2 and len(prev20)>=10 and len(rec20)>=10:
        mig=float(rec20.close.median()/prev20.close.median()-1); lowrise=float(rec20.low.min()/prev20.low.min()-1)
        stall=abs(mig)<=.05 and lowrise<=.02
    return dict(score=float(s),event=ei,abs=100*a,storage=storage,stall=stall)

def lows(d,ci):
    out=[]
    for i in range(max(21,ci-119),ci+1):
        if float(d.loc[i,"low"])<=float(d.loc[i-20:i-1,"low"].min())*1.02:
            if not out or (d.loc[i,"date"]-d.loc[out[-1],"date"]).days>=7: out.append(i)
    return out[-3:]

def S(d,ci):
    ev=lows(d,ci)
    if len(ev)<2:return dict(score=0,events=ev,lowdef=0,reb=0,flow=0,dry=0,toe=False)
    ls=[float(d.loc[i,"low"]) for i in ev]; med=np.median(ls); ld=clip(100-50*(max(ls)-min(ls))/med)
    def de(i):
        j=max(0,i-3); ret=min(float(d.loc[i,"close"]/d.loc[j,"close"]-1),0); vr=float(d.loc[i,"vr20"]) if pd.notna(d.loc[i,"vr20"]) and d.loc[i,"vr20"]>0 else 1
        return abs(ret)/vr
    do,dn=de(ev[0]),de(ev[-1]); q=clip(100*(1-dn/do)) if do>0 else 0
    v0,v1=float(d.loc[ev[0],"volume"]),float(d.loc[ev[-1],"volume"]); dry=clip(100*(1-v1/v0)) if v0 else 0
    rebs=[]
    for i in ev:
        f=d.loc[i+1:min(ci,i+3)]
        if len(f): rebs.append(float(f.high.max()/d.loc[i,"close"]-1))
    rb=clip(100*(np.median(rebs) if rebs else 0)/.20)
    old,new=ev[-2],ev[-1]; flow=(50 if d.loc[new,"obv"]>d.loc[old,"obv"] else 0)+(50 if pd.notna(d.loc[new,"cmf20"]) and pd.notna(d.loc[old,"cmf20"]) and d.loc[new,"cmf20"]>d.loc[old,"cmf20"] else 0)
    s=.30*ld+.25*q+.20*dry+.15*rb+.10*flow
    has8=False
    for i in ev:
        if (d.loc[ci,"date"]-d.loc[i,"date"]).days<=60:
            f=d.loc[i+1:min(ci,i+3)]
            if len(f) and float(f.high.max()/d.loc[i,"close"]-1)>=.08: has8=True
    if not has8:s=min(s,64)
    toe=False
    for i in range(max(20,ci-59),ci+1):
        if pd.isna(d.loc[i,"vr20"]) or d.loc[i,"vr20"]<2:continue
        rng=float(d.loc[i,"high"]-d.loc[i,"low"]); upper=(float(d.loc[i,"close"]-d.loc[i,"low"])/rng) if rng>0 else .5
        f=d.loc[i+1:min(ci,i+3)]
        if (d.loc[i,"close"]>d.loc[i,"open"] and upper>=.5) or (len(f) and float(f.high.max()/d.loc[i,"close"]-1)>=.08): toe=True; break
    return dict(score=float(s),events=ev,lowdef=ld,reb=rb,flow=flow,dry=dry,toe=toe)

def R(d,ci):
    Li=tr=None
    for i in range(max(20,ci-179),ci+1):
        if d.loc[i,"low"]<=d.loc[i-20:i-1,"low"].min():
            L=float(d.loc[i,"low"]); hit=d.loc[i+1:ci].index[d.loc[i+1:ci,"high"]>=1.5*L]
            if len(hit):Li=i;tr=int(hit[0]);break
    if Li is None:return dict(score=0,meaning=False,ret=np.nan,rs=0,dry=0,mig=0,ms=20)
    run=-np.inf; hi=tr; ps=None
    for i in range(tr,ci+1):
        if d.loc[i,"high"]>run:run=float(d.loc[i,"high"]);hi=i
        if i+4<=ci and len(d.loc[i:i+4])==5 and (d.loc[i:i+4,"close"]<=.9*run).all():ps=i;break
    if ps is None or ci-hi<5:return dict(score=0,meaning=False,ret=np.nan,rs=0,dry=0,mig=0,ms=20)
    L,H=float(d.loc[Li,"low"]),float(d.loc[hi,"high"]); pb=d.loc[hi+1:ci]
    if not len(pb) or H<=L:return dict(score=0,meaning=False,ret=np.nan,rs=0,dry=0,mig=0,ms=20)
    pi=int(pb.low.idxmin()); P=float(d.loc[pi,"low"]); ret=(P-L)/(H-L); rs=0 if ret<.3 else 40 if ret<.5 else 70 if ret<.7 else 100
    vi=float(d.loc[Li:hi,"volume"].mean()); vp=float(pb.volume.mean()); dry=clip(100*(1-vp/vi)) if vi else 0
    den=abs(float(d.loc[hi,"obv"]-d.loc[Li,"obv"])); ob=clip(100*(1-(d.loc[hi,"obv"]-d.loc[pi,"obv"])/den)) if den else 0
    cm=(50 if pd.notna(d.loc[ci,"cmf20"]) and pd.notna(d.loc[pi,"cmf20"]) and d.loc[ci,"cmf20"]>d.loc[pi,"cmf20"] else 0)+(50 if pd.notna(d.loc[ci,"cmf20"]) and d.loc[ci,"cmf20"]>0 else 0)
    old=d.loc[max(0,ci-79):max(0,ci-20)]; new=d.loc[max(0,ci-19):ci]; mig=float(new.close.median()/old.close.median()-1) if len(old) else 0; ms=mscore(mig)
    s=.4*rs+.2*dry+.15*ob+.15*cm+.1*ms
    if ret<.3:s=min(s,59)
    return dict(score=float(s),meaning=True,ret=ret,rs=rs,dry=dry,mig=mig,ms=ms)

def classify(sym,d,cut,fwd):
    d=indicators(d); ix=d.index[d.date<=cut]
    if not len(ix) or ix[-1]<180:return {"symbol":sym,"status":"insufficient_pre","class":None}
    ci=int(ix[-1]); pre=d.loc[:ci]; e,s,r=E(pre,ci),S(pre,ci),R(pre,ci)
    rec=pre.loc[max(0,ci-59):ci]; prv=pre.loc[max(0,ci-119):max(0,ci-60)]
    dd=float(rec.low.min()/prv.low.min()-1) if len(prv) else -1; floor=100 if dd>=.1 else 75 if dd>=-.05 else 40 if dd>=-.1 else 0
    absorb=e["abs"] if e["event"] is not None else s["lowdef"]*min(1,s["reb"]/50) if len(s["events"])>=2 else 0
    flow=s["flow"]; dry=r["dry"] if r["meaning"] else s["dry"]; G=.4*floor+.3*absorb+.2*flow+.1*dry
    r90=float(pre.loc[ci,"close"]/pre.loc[ci-90,"close"]-1); P=pscore(r90)
    if r["score"]>=max(e["score"],s["score"]) and r["meaning"] and r["ret"]>=.7:P=P+(100-P)/2
    eng=max(e["score"],s["score"],r["score"]); T=eng*(.4+.6*P/100)*(.5+.5*G/100)
    rsafe=r["rs"] if r["meaning"] else 50; ma=[pre.loc[ci,"ma20"],pre.loc[ci,"ma60"],pre.loc[ci,"ma120"]]
    comp=cscore((max(ma)-min(ma))/pre.loc[ci,"close"]) if all(pd.notna(x) for x in ma) else 0
    safety=.25*absorb+.20*floor+.10*dry+.15*rsafe+.15*r["ms"]+.05*comp+.10*flow
    strong="E" if e["score"]>=max(s["score"],r["score"]) else "S" if s["score"]>=r["score"] else "R"
    if strong=="E": trans=e["storage"] and not e["stall"]; state="E_STORAGE_PASS" if trans else ("E_STALL" if e["stall"] else "E_STORAGE_FAIL")
    elif strong=="S": trans=s["toe"]; state="S_TO_E_PASS" if trans else "S_READY_E_WAIT"
    else: trans=r["meaning"] and r["ret"]>=.3; state="R_PASS" if trans else "R_FAIL"
    fire=T>=70 and safety>=55 and G>=65 and trans
    cls="FIRE" if fire else "SAFE" if safety>=55 and T>=40 else "REJECT"
    cc=float(pre.loc[ci,"close"]); fut=d[(d.date>cut)&(d.date<=cut+pd.Timedelta(days=fwd))]; fx=float(fut.high.max()/cc) if len(fut) else np.nan
    return {"symbol":sym,"status":"ok","cutoff":str(cut.date()),"cutoff_close":cc,"E":round(e["score"],2),"S":round(s["score"],2),"R":round(r["score"],2),"G":round(G,2),"safety":round(safety,2),"T_score":round(T,2),"P":round(P,2),"engine":strong,"state":state,"transition_pass":bool(trans),"class":cls,"future_max":round(fx,4) if pd.notna(fx) else None,"hit_2x":bool(pd.notna(fx) and fx>=2),"hit_3x":bool(pd.notna(fx) and fx>=3),"hit_5x":bool(pd.notna(fx) and fx>=5),"hit_10x":bool(pd.notna(fx) and fx>=10)}

def wilson(k,n,z=1.95996398454):
    if not n:return (np.nan,np.nan)
    p=k/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,ctr-half),min(1,ctr+half)

def main():
    a=argparse.ArgumentParser(); a.add_argument("--cutoff",required=True); a.add_argument("--symbols",default="universe_100.txt"); a.add_argument("--future-days",type=int,default=180); a.add_argument("--outdir",default="results"); x=a.parse_args()
    cut=pd.Timestamp(x.cutoff); start=cut-pd.Timedelta(days=420); end=cut+pd.Timedelta(days=x.future_days)
    syms=[s.strip().upper() for s in Path(x.symbols).read_text().splitlines() if s.strip() and not s.startswith("#")]
    out=Path(x.outdir); out.mkdir(exist_ok=True); rows=[]
    print("Data source priority: Binance official public monthly ZIP -> data-api.binance.vision recent-tail fallback")
    preflight_public_zip(start,end)
    print("Source stats (initial):",SOURCE_STATS)
    for n,s in enumerate(syms,1):
        print(f"[{n}/{len(syms)}] {s}",flush=True)
        try:
            d,source=fetch(s,start,end)
            rec=classify(s,d,cut,x.future_days) if len(d)>=200 else {"symbol":s,"status":"insufficient_data","class":None}
            rec["data_source"]=source
            rows.append(rec)
        except Exception as e:
            rows.append({"symbol":s,"status":"error","class":None,"data_source":None,"error":str(e)[:500]})
        finally:
            time.sleep(SYMBOL_THROTTLE_SECONDS)
    print("Source stats (final):",SOURCE_STATS)
    df=pd.DataFrame(rows)
    if "class" not in df.columns: df["class"]=None
    if "data_source" not in df.columns: df["data_source"]=None
    print("DataFrame columns:",list(df.columns))
    print("Status counts:")
    print(df["status"].value_counts(dropna=False).to_string())
    ok_count=int((df["status"]=="ok").sum())
    error_count=int((df["status"]=="error").sum())
    total_universe=len(syms)
    ok_rate=ok_count/total_universe if total_universe else 0.0
    print(f"RUN SELF-CHECK: total_universe={total_universe} ok_count={ok_count} error_count={error_count} ok_rate={ok_rate:.1%}")
    errors=df[df["status"]=="error"]
    if len(errors):
        print("Representative errors (up to 10):")
        cols=[c for c in ["symbol","data_source","error"] if c in errors.columns]
        print(errors[cols].head(10).to_string(index=False))
    pred=[c for c in df.columns if not c.startswith("future_") and not c.startswith("hit_")]
    df[pred].to_csv(out/"frozen_predictions.csv",index=False); df.to_csv(out/"outcomes.csv",index=False)

    if ok_count<=0:
        raise RuntimeError("SELF-CHECK FAILED: status=='ok' must be > 0; inspect source stats and errors above.")
    if "class" not in df.columns:
        raise RuntimeError("SELF-CHECK FAILED: DataFrame is missing class column.")
    v=df[df.status=="ok"].copy()
    if "future_max" not in v.columns or not pd.to_numeric(v["future_max"],errors="coerce").notna().any():
        raise RuntimeError("SELF-CHECK FAILED: future multiple (future_max) is not calculable for any ok row.")

    base=float(v.hit_5x.mean()) if len(v) else np.nan; summ=[]
    for cls in ["FIRE","SAFE","REJECT"]:
        g=v[v["class"]==cls]
        if not len(g):continue
        k=int(g.hit_5x.sum()); lo,hi=wilson(k,len(g)); k10=int(g.hit_10x.sum()); l10,h10=wilson(k10,len(g))
        summ.append({"class":cls,"n":len(g),"median_future_x":round(float(g.future_max.median()),3),"mean_future_x":round(float(g.future_max.mean()),3),"rate_2x":round(float(g.hit_2x.mean()),4),"rate_3x":round(float(g.hit_3x.mean()),4),"rate_5x":round(k/len(g),4),"wilson5_low":round(lo,4),"wilson5_high":round(hi,4),"rate_10x":round(k10/len(g),4),"wilson10_low":round(l10,4),"wilson10_high":round(h10,4),"enrichment5":round((k/len(g))/base,3) if base and base>0 else None})
    summary_df=pd.DataFrame(summ)
    if summary_df.empty or "class" not in summary_df.columns:
        raise RuntimeError("SELF-CHECK FAILED: summary could not be generated from ok rows.")
    summary_df.to_csv(out/"summary.csv",index=False)
    print("SELF-CHECK PASS: ok rows > 0, class present, future multiple calculable, summary generated")
    print(summary_df.to_string(index=False))
    if ok_rate<0.90:
        raise RuntimeError(f"DATA VALIDATION FAILED: ok rate {ok_rate:.1%} is below required 90.0% ({ok_count}/{total_universe}).")

if __name__=="__main__": main()
