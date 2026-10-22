# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained local review UI; no remote scripts or raw prompt payloads."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Report


def write_preview(report: Report, path: Path) -> None:
    data = json.dumps(report.to_dict(), allow_nan=False).replace("<", "\\u003c")
    html = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Standard evaluation analyses</title><style>
body{font:15px system-ui;color:#203044;background:#f4f6f9;margin:0}main{max-width:1200px;margin:auto;padding:32px}
h1{font-size:29px;margin:0 0 8px}header{margin-bottom:26px}p{line-height:1.55}select,input,button{font:inherit;padding:9px;border:1px solid #bac5d3;border-radius:6px;background:white}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.cards{display:flex;gap:10px;flex-wrap:wrap}.card{padding:12px 16px;border:1px solid #dbe1e8;background:white;border-radius:8px;min-width:150px}.card b{display:block;font-size:21px}.card span{font-size:12px;color:#566578}.good{color:#207052}.partial{color:#946100}.unavailable{color:#8b3540}.panel{background:white;border:1px solid #dbe1e8;border-radius:9px;padding:18px;margin-top:18px}.table{overflow:auto;max-height:540px}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e7ebef}th{position:sticky;top:0;background:#eaf0f7}li{line-height:1.5;margin-bottom:6px}svg{width:100%;max-height:300px}small{color:#566578}button{cursor:pointer}#availability{margin-bottom:20px}
</style><main><header><h1>Standard evaluation analyses</h1><div id="cohort"></div><p>Computed locally from captured calls and scored rollouts. Model-call time and observed excess are descriptive measurements.</p></header>
<div id="availability" class="cards"></div><div class="controls"><select id="analysis" aria-label="Analysis"></select><input id="filter" placeholder="Filter displayed table rows" aria-label="Filter table"><button id="download">Download results JSON</button></div>
<h2 id="title"></h2><p id="description"></p><div id="metrics" class="cards"></div><div class="panel" id="chart"></div><div class="panel"><b>Interpretation and coverage</b><ul id="notes"></ul><div id="coverage"></div></div><div class="panel"><p id="rows"></p><div class="table"><table><thead id="head"></thead><tbody id="body"></tbody></table></div></div></main>
<script>const data=__DATA__;
const $=id=>document.getElementById(id),fmt=v=>v===null?'unknown':typeof v==='number'?Number(v.toFixed(4)).toLocaleString():String(v);
$('cohort').textContent=Object.entries(data.cohort).map(([k,v])=>`${k}: ${v}`).join(' · ');
for(const status of ['available','partial','unavailable']){let el=document.createElement('div');el.className='card '+status;el.textContent=`${data.results.filter(r=>r.status===status).length} ${status}`;$('availability').appendChild(el)}
for(const r of data.results){const op=document.createElement('option');op.value=r.name;op.textContent=r.name+' — '+r.status;$('analysis').appendChild(op)}
function card(key,value){let el=document.createElement('div');el.className='card';let b=document.createElement('b');b.textContent=fmt(value);let span=document.createElement('span');span.textContent=key;el.append(b,span);return el}
function draw(r){const table=r.tables.detail;let bars=[];if(r.name==='long_calls')bars=table.rows.filter(x=>x[7]!==null).map(x=>[x[0]+' >'+x[1],x[7]]);if(r.name==='repeat_excess')bars=table.rows.filter(x=>x[0]==='all_scored').slice(0,12).map(x=>[x[1]+'/'+x[2],x[7]]);$('chart').replaceChildren();if(!bars.length){$('chart').textContent='Inspect the typed measurements below.';return}const heading=document.createElement('h3');heading.textContent=r.name==='long_calls'?'Share of measured model-call time':'Observed excess model-call seconds';$('chart').appendChild(heading);const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 1000 280');const max=Math.max(...bars.map(x=>x[1]),.001);bars.forEach(([label,value],i)=>{const x=10+i*980/bars.length,w=850/bars.length,h=value/max*210;const rect=document.createElementNS(ns,'rect');for(const [k,v]of Object.entries({x,y:225-h,width:w,height:h,fill:'#3979b8'}))rect.setAttribute(k,v);const title=document.createElementNS(ns,'title');title.textContent=label+': '+fmt(value);rect.appendChild(title);svg.appendChild(rect);const valueText=document.createElementNS(ns,'text');valueText.setAttribute('x',x);valueText.setAttribute('y',Math.max(12,217-h));valueText.setAttribute('font-size',13);valueText.textContent=r.name==='long_calls'?(value*100).toFixed(1)+'%':fmt(value)+' s';svg.appendChild(valueText);const text=document.createElementNS(ns,'text');text.setAttribute('x',x);text.setAttribute('y',246);text.setAttribute('font-size',11);text.textContent=label;svg.appendChild(text)});$('chart').appendChild(svg)}
function render(){const r=data.results.find(r=>r.name===$('analysis').value);$('title').textContent=r.name+' · '+r.status;$('description').textContent=r.description;$('metrics').replaceChildren(...Object.entries(r.metrics).map(([k,v])=>card(k,v)));$('notes').replaceChildren();for(const note of [r.reason,...r.notes].filter(Boolean)){let el=document.createElement('li');el.textContent=note;$('notes').appendChild(el)}$('coverage').textContent=Object.entries(r.coverage).map(([k,v])=>k+': '+v).join(' · ');let t=r.tables.detail,q=$('filter').value.toLowerCase(),filtered=t.rows.filter(row=>row.some(v=>String(v).toLowerCase().includes(q)));$('rows').textContent=`${filtered.length} matching rows · ${t.rows.length} retained / ${t.total_rows} computed${t.truncated?' · DISPLAY TRUNCATED; summary metrics use the full cohort':''}`;$('head').replaceChildren();let tr=document.createElement('tr');for(const c of t.columns){let th=document.createElement('th');th.textContent=c;tr.appendChild(th)}$('head').appendChild(tr);$('body').replaceChildren();for(const row of filtered.slice(0,500)){let tr=document.createElement('tr');for(const v of row){let td=document.createElement('td');td.textContent=fmt(v);tr.appendChild(td)}$('body').appendChild(tr)}draw(r)}
$('analysis').onchange=render;$('filter').oninput=render;$('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='results.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};if(data.results.length)render();
</script></html>"""
    path.write_text(html.replace("__DATA__", data))
