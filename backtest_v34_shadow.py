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
DEVELOPMENT_CUTOFF="2023-10-15"
DIAGNOSTIC_SYMBOLS=["VETUSDT","ZRXUSDT","GRTUSDT","THETAUSDT","COTIUSDT","RUNEUSDT","FETUSDT","IDUSDT","SOLUSDT","NEARUSDT","AVAXUSDT","INJUSDT"]

SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"crypto-scanner-v3.4-shadow-binance-public-data/1.0"})
SESSION.mount("https://",HTTPAdapter(pool_connections=10,pool_maxsize=10,max_retries=0))
SESSION.mount("http://",HTTPAdapter(pool_connections=10,pool_maxsize=10,max_retries=0))

def clip(x,a=0,b=100): return float(max(a,min(b,0 if pd.isna(x) else x)))
def tsms(x): return int(pd.Timestamp(x).timestamp()*1000)

def _parse_open_time(s):
    v=pd.to_numeric(s,errors="coerce")
    out=pd.Series(pd.NaT,index=v.index,dtype="datetime64[ns]")
    ms=v.notna() & (v.abs()<1e14); us=v.notna() & ~ms
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
    d=d[d["open_time"].notna()].copy(); d["date"]=_parse_open_time(d["open_time"])
    return _normalize_frame(d)

def _month_cache_paths(sym,period):
    p=CACHE/"monthly"/sym; p.mkdir(parents=True,exist_ok=True)
    stem=f"{sym}-1d-{period.year:04d}-{period.month:02d}"
    return p/f"{stem}.zip",p/f"{stem}.csv"

def _month_url(sym,period):
    name=f"{sym}-1d-{period.year:04d}-{period.month:02d}.zip"
    return f"{PUBLIC_BASE}/data/spot/monthly/klines/{sym}/1d/{name}"

def _sleep_backoff(attempt): time.sleep(BACKOFF_BASE*(2**(attempt-1)))

def _request_with_retry(url,*,params=None,timeout=30,sym=None,month=None,max_retries=MAX_RETRIES):
    last_error=None
    for attempt in range(1,max_retries+1):
        try:
            r=SESSION.get(url,params=params,timeout=timeout)
            if r.status_code==404:return r,attempt
            if 500<=r.status_code<=599: raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}",response=r)
            if not r.ok: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r,attempt
        except (requests.exceptions.ConnectionError,requests.exceptions.Timeout,requests.exceptions.ChunkedEncodingError,requests.exceptions.HTTPError) as e:
            last_error=e; label=str(month) if month is not None else "n/a"
            print(f"DOWNLOAD_RETRY symbol={sym or 'n/a'} month={label} retry_count={attempt} error={type(e).__name__}: {e}",flush=True)
            if attempt<max_retries:_sleep_backoff(attempt)
    label=str(month) if month is not None else "n/a"
    print(f"DOWNLOAD_FAILED symbol={sym or 'n/a'} month={label} retry_count={max_retries} error={type(last_error).__name__}: {last_error}",flush=True)
    raise RuntimeError(f"DOWNLOAD_FAILED symbol={sym or 'n/a'} month={label} retry_count={max_retries} error={last_error}") from last_error

def fetch_public_month(sym,period,count_stats=True,use_cache=True):
    zip_path,csv_path=_month_cache_paths(sym,period); url=_month_url(sym,period)
    if use_cache and csv_path.exists():
        try:
            d=_normalize_frame(pd.read_csv(csv_path,parse_dates=["date"]))
            if not d.empty:
                if count_stats: SOURCE_STATS["public_zip_success"]+=1
                return d
        except Exception: csv_path.unlink(missing_ok=True)
    if use_cache and zip_path.exists():
        try:
            d=_parse_zip_bytes(zip_path.read_bytes())
            if not d.empty:
                d.to_csv(csv_path,index=False)
                if count_stats: SOURCE_STATS["public_zip_success"]+=1
                return d
        except Exception:
            zip_path.unlink(missing_ok=True); csv_path.unlink(missing_ok=True)
    r,attempts=_request_with_retry(url,timeout=30,sym=sym,month=period)
    if r.status_code==404:
        if count_stats: SOURCE_STATS["public_zip_missing"]+=1
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    d=_parse_zip_bytes(r.content)
    if d.empty: raise RuntimeError(f"PUBLIC_ZIP_EMPTY_AFTER_PARSE symbol={sym} month={period} retry_count={attempts}")
    zip_path.write_bytes(r.content); d.to_csv(csv_path,index=False)
    if count_stats: SOURCE_STATS["public_zip_success"]+=1
    return d

def fetch_market_api(sym,start,end):
    rows=[]; cur=pd.Timestamp(start)
    try:
        while cur<=end:
            r,_=_request_with_retry(MARKET_DATA_HOST+"/api/v3/klines",
                params=dict(symbol=sym,interval="1d",startTime=tsms(cur),endTime=tsms(end+pd.Timedelta(days=1))-1,limit=1000),
                timeout=20,sym=sym,month=f"market_api:{cur.date()}")
            if r.status_code==404: raise RuntimeError("HTTP 404")
            b=r.json()
            if not b: break
            rows += [[x[0],x[1],x[2],x[3],x[4],x[5]] for x in b]
            nxt=pd.to_datetime(b[-1][0],unit="ms")+pd.Timedelta(days=1)
            if nxt<=cur: break
            cur=nxt; time.sleep(.05)
        if not rows: raise RuntimeError("NO_DATA")
        d=pd.DataFrame(rows,columns=["open_time","open","high","low","close","volume"])
        d["date"]=pd.to_datetime(pd.to_numeric(d["open_time"],errors="coerce"),unit="ms",errors="coerce")
        d=_normalize_frame(d); d=d[(d.date>=start)&(d.date<=end)].copy()
        if d.empty: raise RuntimeError("NO_DATA_IN_RANGE")
        SOURCE_STATS["market_api_success"]+=1; return d
    except Exception:
        SOURCE_STATS["market_api_failed"]+=1; raise

def fetch(sym,start,end):
    months=pd.period_range(start=pd.Timestamp(start).to_period("M"),end=pd.Timestamp(end).to_period("M"),freq="M")
    parts=[]; missing=[]
    for period in months:
        d=fetch_public_month(sym,period)
        if d.empty: missing.append(period)
        else: parts.append(d)
    used_api=False; now=pd.Timestamp.utcnow().tz_localize(None); recent_floor=(now-pd.Timedelta(days=62)).to_period("M")
    recent_missing=[p for p in missing if p>=recent_floor]
    if recent_missing:
        gap_start=max(pd.Timestamp(start),recent_missing[0].start_time)
        if parts:
            last=max(x.date.max() for x in parts); gap_start=max(gap_start,last+pd.Timedelta(days=1))
        if gap_start<=end:
            try:
                parts.append(fetch_market_api(sym,gap_start,end)); used_api=True
            except Exception: pass
    if not parts:return pd.DataFrame(columns=["date","open","high","low","close","volume"]),"binance_public_zip"
    d=pd.concat(parts,ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    d=d[(d.date>=start)&(d.date<=end)].copy().reset_index(drop=True)
    return d,"binance_market_data_api" if used_api else "binance_public_zip"

def preflight_public_zip(start,end):
    months=list(pd.period_range(start=pd.Timestamp(start).to_period("M"),end=pd.Timestamp(end).to_period("M"),freq="M"))
    now_month=pd.Timestamp.utcnow().tz_localize(None).to_period("M"); candidates=[p for p in months if p<now_month]
    if len(candidates)<2: raise RuntimeError("PRE-FLIGHT FAILED: fewer than two completed months exist in requested range")
    test_months=candidates[:2]; symbols=["BTCUSDT","ETHUSDT","ZRXUSDT"]; total=len(symbols)*len(test_months); passed=0; failures=[]
    print(f"PRE-FLIGHT TEST: symbols={symbols} months={[str(p) for p in test_months]} tests={total}")
    for sym in symbols:
        for period in test_months:
            try:
                d=fetch_public_month(sym,period,count_stats=False,use_cache=True)
                if d.empty: raise RuntimeError("ZIP missing/404 or empty")
                if len(d)<20: raise RuntimeError(f"parsed rows={len(d)} (<20)")
                passed+=1; print(f"PRE-FLIGHT PASS symbol={sym} month={period} rows={len(d)}",flush=True)
            except Exception as e:
                failures.append(f"{sym} {period}: {e}"); print(f"PRE-FLIGHT FAIL symbol={sym} month={period} error={e}",flush=True)
    rate=passed/total if total else 0; print(f"PRE-FLIGHT SUCCESS RATE: {passed}/{total} = {rate:.1%}")
    if passed<total: raise RuntimeError("PRE-FLIGHT FAILED: "+" | ".join(failures[:6]))
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

def pscore(r): return 100 if r<=.2 else 85 if r<=.5 else 65 if r<=1 else 35 if r<=2 else 15 if r<=3 else 0
def epscore(r): return 100 if r<=.1 else 80 if r<=.3 else 60 if r<=.6 else 35 if r<=1 else 15 if r<=2 else 0
def mscore(m): return 20 if m<=0 else 60 if m<=.1 else 100 if m<=.3 else 70 if m<=.6 else 35 if m<=1 else 0
def cscore(x): return 100 if x<=.05 else 75 if x<=.1 else 50 if x<=.2 else 25 if x<=.3 else 0

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

def classify_v33(sym,d,cut,fwd):
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
    if strong=="E": trans=e["storage"] and not e["stall"]
    elif strong=="S": trans=s["toe"]
    else: trans=r["meaning"] and r["ret"]>=.3
    fire=T>=70 and safety>=55 and G>=65 and trans
    cls="FIRE" if fire else "SAFE" if safety>=55 and T>=40 else "REJECT"
    cc=float(pre.loc[ci,"close"]); fut=d[(d.date>cut)&(d.date<=cut+pd.Timedelta(days=fwd))]; fx=float(fut.high.max()/cc) if len(fut) else np.nan
    return {"symbol":sym,"status":"ok","cutoff":str(cut.date()),"cutoff_close":cc,"E":round(e["score"],2),"S":round(s["score"],2),"R":round(r["score"],2),"G":round(G,2),"safety":round(safety,2),"T_score":round(T,2),"P":round(P,2),"class":cls,"future_max":round(fx,4) if pd.notna(fx) else None}

def block_features(d,ci):
    w=d.loc[ci-179:ci].copy().reset_index(drop=True)
    if len(w)<180:return []
    obv_span=max(float(w.obv.max()-w.obv.min()),1e-12); obv0=float(w.obv.iloc[0]); out=[]
    for b in range(6):
        x=w.iloc[b*30:(b+1)*30]; down=[]; up=[]
        for j in range(1,len(x)):
            prev=float(x.close.iloc[j-1]); cur=float(x.close.iloc[j]); vr=float(x.vr20.iloc[j]) if pd.notna(x.vr20.iloc[j]) and x.vr20.iloc[j]>0 else 1
            rr=cur/prev-1
            if rr<0: down.append(abs(rr)/vr)
            elif rr>0: up.append(rr/vr)
        out.append({"block":b+1,"median_close":float(x.close.median()),"low_q10":float(x.close.quantile(.10)),"median_volume":float(x.volume.median()),"abnormal_volume_count":int((x.vr20>=2).sum()),"normalized_obv":float((x.obv.median()-obv0)/obv_span),"median_cmf20":float(x.cmf20.median()) if x.cmf20.notna().any() else np.nan,"downside_efficiency":float(np.median(down)) if down else 0.0,"upside_response_efficiency":float(np.median(up)) if up else 0.0})
    return out

def trajectory(sym,d,cut,fwd):
    ind=indicators(d); ix=ind.index[ind.date<=cut]
    if not len(ix) or ix[-1]<180:return {"status":"insufficient_pre"}
    ci=int(ix[-1]); blocks=block_features(ind,ci)
    if len(blocks)!=6:return {"status":"insufficient_blocks"}
    q=np.array([x["low_q10"] for x in blocks]); downs=np.array([x["downside_efficiency"] for x in blocks]); ups=np.array([x["upside_response_efficiency"] for x in blocks])
    floor_migration=float(q[-1]/q[0]-1) if q[0] else np.nan; floor_recent=float(q[-1]/q[-2]-1) if q[-2] else np.nan
    early_down=float(np.mean(downs[:2])); late_down=float(np.mean(downs[-2:])); sell_pressure_decay=float(1-late_down/(early_down+1e-12))
    early_up=float(np.mean(ups[:2])); late_up=float(np.mean(ups[-2:])); demand_response=float(late_up/(early_up+1e-12)-1) if early_up>0 else (1.0 if late_up>0 else 0.0)
    shock_idx=[i for i in range(max(20,ci-179),ci+1) if pd.notna(ind.loc[i,"vr20"]) and ind.loc[i,"vr20"]>=2 and ind.loc[i,"close"]>=ind.loc[i,"open"]]
    stor=[]; levels=[]; responses=[]
    for i in shock_idx:
        base=float(ind.loc[max(0,i-5):i-1,"close"].median()) if i>0 else float(ind.loc[i,"close"]); p7=ind.loc[i+1:min(ci,i+7)]; p14=ind.loc[i+1:min(ci,i+14)]
        if len(p7): responses.append(float(p7.high.max()/ind.loc[i,"close"]-1))
        vals=[]
        if len(p7): vals.append(float(p7.close.iloc[-1]/base-1))
        if len(p14): vals.append(float(p14.close.iloc[-1]/base-1))
        if vals: stor.append(float(np.mean(vals)))
        levels.append(float(ind.loc[i,"close"]))
    event_price_storage=float(np.median(stor)) if stor else np.nan; demand_event_response=float(np.median(responses)) if responses else 0.0
    price_level_migration=float(levels[-1]/levels[0]-1) if len(levels)>=2 and levels[0] else 0.0
    pre=ind.loc[:ci]; r=R(pre,ci); retention=float(r["ret"]) if r["meaning"] and pd.notna(r["ret"]) else np.nan; pullback_dryup=float(r["dry"]/100) if r["meaning"] else 0.0
    r90=float(pre.loc[ci,"close"]/pre.loc[ci-90,"close"]-1); asymmetry=float(pscore(r90)/100)
    recent90=pre.loc[max(20,ci-89):ci]; base_low=float(pre.loc[max(0,ci-179):ci].low.quantile(.10)); highvol_newlow=int(((recent90.vr20>=2)&(recent90.low<=base_low)).sum()); abnormal_recent=sum(x["abnormal_volume_count"] for x in blocks[-2:])
    veto=[]
    if abnormal_recent>=2 and price_level_migration<=0:veto.append("REPEATED_ABNORMAL_VOLUME_NO_PRICE_MIGRATION")
    if len(stor)>=2 and np.median(stor)<=0:veto.append("VOLUME_SHOCK_RETURNS_TO_BASE")
    if pd.notna(retention) and retention<.30:veto.append("RETENTION_HARD_FAIL_LT30")
    if highvol_newlow>=2:veto.append("REPEATED_HIGH_VOLUME_NEW_LOW")
    if floor_recent<-.05 and floor_migration<0:veto.append("RECENT_FLOOR_DECLINE")
    if abnormal_recent>=2 and demand_event_response<=0:veto.append("VOLUME_WITHOUT_DEMAND_RESPONSE")
    if abnormal_recent>=2 and abs(floor_migration)<=.02 and abs(price_level_migration)<=.02 and (pd.isna(event_price_storage) or event_price_storage<=0):veto.append("STRUCTURAL_STALL")
    if asymmetry<=.15:veto.append("PRICE_OVER_REFLECTED")
    supply_axis="PASS" if ((floor_migration>0 and sell_pressure_decay>0) or (floor_recent>=0 and sell_pressure_decay>0)) else ("FAIL" if floor_migration<0 and sell_pressure_decay<0 else "NEUTRAL")
    demand_axis="PASS" if (demand_event_response>0 and (pd.isna(event_price_storage) or event_price_storage>0)) else ("FAIL" if demand_event_response<=0 and (not pd.isna(event_price_storage) and event_price_storage<=0) else "NEUTRAL")
    migration_axis="PASS" if ((price_level_migration>0 and (pd.isna(retention) or retention>=.5)) or (pd.notna(retention) and retention>.7 and pullback_dryup>0)) else ("FAIL" if ((pd.notna(retention) and retention<.3) or (price_level_migration<0 and floor_recent<0)) else "NEUTRAL")
    asymmetry_axis="PASS" if asymmetry>=.65 else ("NEUTRAL" if asymmetry>=.35 else "FAIL")
    axes=[supply_axis,demand_axis,migration_axis,asymmetry_axis]; passes=sum(x=="PASS" for x in axes)
    if veto or supply_axis=="FAIL" or demand_axis=="FAIL": v34_class="REJECT"; state="VETO_OR_CORE_FAIL"
    elif all(x=="PASS" for x in axes): v34_class="FIRE"; state="ALL_AXES_PASS"
    elif supply_axis=="PASS" and demand_axis=="PASS" and migration_axis in ("PASS","NEUTRAL") and asymmetry_axis in ("PASS","NEUTRAL") and not (pd.notna(retention) and retention<.30): v34_class="SAFE"; state="STRUCTURE_APPROVED"
    elif passes>=2: v34_class="WATCH"; state="CONFIRMATION_PENDING"
    else: v34_class="REJECT"; state="INSUFFICIENT_STRUCTURE"
    rank_base={"FIRE":400,"SAFE":300,"WATCH":200,"REJECT":100}[v34_class]; v34_rank=rank_base+passes*10+(5 if floor_migration>0 else 0)+(5 if demand_event_response>0 else 0)+(5 if price_level_migration>0 else 0)+asymmetry*5
    cc=float(pre.loc[ci,"close"]); fut=ind[(ind.date>cut)&(ind.date<=cut+pd.Timedelta(days=fwd))]; fx=float(fut.high.max()/cc) if len(fut) else np.nan
    return {"status":"ok","blocks":blocks,"floor_migration":floor_migration,"sell_pressure_decay":sell_pressure_decay,"demand_response":demand_event_response,"event_price_storage":event_price_storage,"price_level_migration":price_level_migration,"pullback_dryup":pullback_dryup,"retention":retention,"asymmetry":asymmetry,"supply_axis":supply_axis,"demand_axis":demand_axis,"migration_axis":migration_axis,"asymmetry_axis":asymmetry_axis,"veto_count":len(veto),"veto_reasons":"|".join(veto),"v34_state":state,"v34_class":v34_class,"v34_rank":v34_rank,"future_multiple":fx,"future_max":fx,"market_regime":"UNKNOWN"}

def summarize(df):
    ok=df[df.status=="ok"].copy(); bases={m:float((ok.future_multiple>=m).mean()) for m in [1.5,2,3,5,10]}; rows=[]
    for cls in ["FIRE","SAFE","WATCH"]:
        g=ok[ok.v34_class==cls].copy()
        if not len(g):continue
        r={"scope":cls,"count":len(g),"median_future_multiple":float(g.future_multiple.median()),"mean_future_multiple":float(g.future_multiple.mean())}
        for m in [1.5,2,3,5,10]:r[f"rate_{str(m).replace('.','_')}x"]=float((g.future_multiple>=m).mean())
        if cls=="FIRE":
            ranked=g.sort_values("v34_rank",ascending=False); top1=ranked.head(1); top3=ranked.head(3); r["fire_5x_precision"]=r["rate_5x"]; r["fire_10x_precision"]=r["rate_10x"]; r["top1_future_multiple"]=float(top1.future_multiple.iloc[0]) if len(top1) else np.nan; r["top1_hit_5x"]=bool(len(top1) and top1.future_multiple.iloc[0]>=5); r["top1_hit_10x"]=bool(len(top1) and top1.future_multiple.iloc[0]>=10); r["top3_5x_rate"]=float((top3.future_multiple>=5).mean()) if len(top3) else np.nan; r["top3_10x_rate"]=float((top3.future_multiple>=10).mean()) if len(top3) else np.nan; r["enrichment_5x"]=r["rate_5x"]/bases[5] if bases[5]>0 else np.nan; r["enrichment_10x"]=r["rate_10x"]/bases[10] if bases[10]>0 else np.nan
        elif cls=="SAFE":
            r["safe_2x_precision"]=r["rate_2x"]; r["safe_3x_precision"]=r["rate_3x"]; r["safe_failure_rate"]=float((g.future_multiple<1.5).mean()); r["enrichment_2x"]=r["rate_2x"]/bases[2] if bases[2]>0 else np.nan; r["enrichment_3x"]=r["rate_3x"]/bases[3] if bases[3]>0 else np.nan
        rows.append(r)
    overall={"scope":"OVERALL","count":len(ok),"median_future_multiple":float(ok.future_multiple.median()),"mean_future_multiple":float(ok.future_multiple.mean())}
    for m in [1.5,2,3,5,10]:overall[f"base_rate_{str(m).replace('.','_')}x"]=bases[m]
    for tag,col,classes in [("v33","v33_class",["FIRE","SAFE","REJECT"]),("v34","v34_class",["FIRE","SAFE","WATCH","REJECT"])]:
        for cls in classes:
            g=ok[ok[col]==cls]; overall[f"{tag}_{cls.lower()}_count"]=len(g)
            if len(g):
                overall[f"{tag}_{cls.lower()}_2x_rate"]=float((g.future_multiple>=2).mean()); overall[f"{tag}_{cls.lower()}_3x_rate"]=float((g.future_multiple>=3).mean()); overall[f"{tag}_{cls.lower()}_5x_rate"]=float((g.future_multiple>=5).mean()); overall[f"{tag}_{cls.lower()}_10x_rate"]=float((g.future_multiple>=10).mean())
    total5=int((ok.future_multiple>=5).sum()); total10=int((ok.future_multiple>=10).sum()); overall["v34_fire_5x_recall"]=float(((ok.v34_class=="FIRE")&(ok.future_multiple>=5)).sum()/total5) if total5 else np.nan; overall["v34_fire_10x_recall"]=float(((ok.v34_class=="FIRE")&(ok.future_multiple>=10)).sum()/total10) if total10 else np.nan
    rows.append(overall); return pd.DataFrame(rows)

def main():
    a=argparse.ArgumentParser(); a.add_argument("--cutoff",required=True); a.add_argument("--symbols",default="universe_100.txt"); a.add_argument("--future-days",type=int,default=180); a.add_argument("--outdir",default="results_v34"); x=a.parse_args()
    cut=pd.Timestamp(x.cutoff); start=cut-pd.Timedelta(days=420); end=cut+pd.Timedelta(days=x.future_days); syms=[s.strip().upper() for s in Path(x.symbols).read_text().splitlines() if s.strip() and not s.startswith("#")]; out=Path(x.outdir); out.mkdir(exist_ok=True); rows=[]
    print("v3.4-shadow precision-first model"); print(f"DEVELOPMENT SET NOTICE: {DEVELOPMENT_CUTOFF} has outcome-seen development status. Do not tune thresholds/rules after inspecting its results."); print("Data source priority: Binance official public monthly ZIP -> data-api.binance.vision recent-tail fallback"); preflight_public_zip(start,end); print("Source stats (initial):",SOURCE_STATS)
    for n,sym in enumerate(syms,1):
        print(f"[{n}/{len(syms)}] {sym}",flush=True)
        try:
            d,source=fetch(sym,start,end)
            if len(d)<200: rows.append({"symbol":sym,"status":"insufficient_data","v34_class":None,"v33_class":None,"data_source":source}); continue
            v33=classify_v33(sym,d,cut,x.future_days); t=trajectory(sym,d,cut,x.future_days)
            if t.get("status")!="ok": rows.append({"symbol":sym,"status":t.get("status"),"v34_class":None,"v33_class":v33.get("class"),"data_source":source}); continue
            rows.append({"symbol":sym,"status":"ok","cutoff":str(cut.date()),"future_max":t["future_max"],"future_multiple":t["future_multiple"],"E":v33["E"],"S":v33["S"],"R":v33["R"],"G":v33["G"],"safety":v33["safety"],"T_score":v33["T_score"],"P":v33["P"],"v33_class":v33["class"],"floor_migration":t["floor_migration"],"sell_pressure_decay":t["sell_pressure_decay"],"demand_response":t["demand_response"],"event_price_storage":t["event_price_storage"],"price_level_migration":t["price_level_migration"],"pullback_dryup":t["pullback_dryup"],"retention":t["retention"],"asymmetry":t["asymmetry"],"supply_axis":t["supply_axis"],"demand_axis":t["demand_axis"],"migration_axis":t["migration_axis"],"asymmetry_axis":t["asymmetry_axis"],"veto_count":t["veto_count"],"veto_reasons":t["veto_reasons"],"v34_state":t["v34_state"],"v34_class":t["v34_class"],"v34_rank":t["v34_rank"],"market_regime":t["market_regime"],"data_source":source})
        except Exception as e: rows.append({"symbol":sym,"status":"error","v34_class":None,"v33_class":None,"data_source":None,"error":str(e)[:500]})
        finally: time.sleep(SYMBOL_THROTTLE_SECONDS)
    print("Source stats (final):",SOURCE_STATS); df=pd.DataFrame(rows); df.to_csv(out/"outcomes_v34.csv",index=False); ok=df[df.status=="ok"].copy()
    if not len(ok):raise RuntimeError("v3.4-shadow produced no ok rows")
    summary=summarize(df); summary.to_csv(out/"summary_v34.csv",index=False); diag=ok[ok.symbol.isin(DIAGNOSTIC_SYMBOLS)].copy(); diag_cols=["symbol","v33_class","v34_class","floor_migration","sell_pressure_decay","demand_response","event_price_storage","price_level_migration","pullback_dryup","retention","asymmetry","supply_axis","demand_axis","migration_axis","asymmetry_axis","veto_count","veto_reasons","future_multiple"]; diag[diag_cols].to_csv(out/"diagnostic_v34.csv",index=False); print("DIAGNOSTIC TABLE (development-set inspection only; no threshold tuning permitted):"); print(diag[diag_cols].to_string(index=False)); print(summary.to_string(index=False))

if __name__=="__main__":main()
