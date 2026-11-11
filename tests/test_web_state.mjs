import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const script=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
function harness() {
  const nodes=new Map();
  function node() {
    const classes=new Set();
    const element={textContent:'',value:'',hidden:false,disabled:false,checked:true,style:{},children:[],events:{},
      classList:{add:(...xs)=>xs.forEach(x=>classes.add(x)),remove:(...xs)=>xs.forEach(x=>classes.delete(x)),contains:x=>classes.has(x),toggle:(x,on)=>on?classes.add(x):classes.delete(x)},
      addEventListener(event,fn){this.events[event]=fn;},setAttribute(){},append(child){this.children.push(child);},
      replaceChildren(){this.children=[];},focus(){}};
    Object.defineProperty(element,'className',{get:()=>[...classes].join(' '),set:value=>{classes.clear();value.split(/\s+/).filter(Boolean).forEach(x=>classes.add(x));}});
    return element;
  }
  for(const [,id] of html.matchAll(/id="([^"]+)"/g))nodes.set(id,node());
  nodes.get('prompt').value='A harmless prompt.';nodes.get('max-tokens').value='64';
  let generation;const intervals=[];let health={ready:true,backend:'gpu',model:'test',session_expires_at:Date.now()/1000-1};
  const context=vm.createContext({document:{getElementById:id=>nodes.get(id),createElement:node,querySelectorAll:()=>[]},
    localStorage:{getItem:()=>null,setItem(){}},URL,AbortSignal,AbortController,TextDecoder,Date,performance,
    setInterval:(fn)=>{intervals.push(fn);return intervals.length;},clearInterval(){},setTimeout,clearTimeout,
    fetch:async url=>url.endsWith('/api/generate')?generation:({ok:true,json:async()=>url.endsWith('/api/health')?health:{examples:[]}})});
  vm.runInContext(script,context);
  return {nodes,intervals,context,setHealth:value=>{health=value;},setGeneration:value=>{generation=value;}};
}
const flush=()=>new Promise(setImmediate);

test('reconnecting clears an old expired session and its visible expiry',async()=>{
  const app=harness();await flush();
  app.intervals[0]();
  assert.equal(app.nodes.get('run-button').disabled,true);
  app.setHealth({ready:true,backend:'gpu',model:'test',session_expires_at:null});
  await app.nodes.get('connect-button').events.click();
  app.intervals[0]();
  assert.equal(app.nodes.get('run-button').disabled,false);
  assert.equal(app.nodes.get('session-status').hidden,true);
  assert.equal(app.nodes.get('connection-status').textContent,'GPU connected');
});

test('interrupted streams distinguish partial output from branches that never ran',async()=>{
  const app=harness();await flush();
  app.setHealth({ready:true,backend:'gpu',model:'test',session_expires_at:null});
  await app.nodes.get('connect-button').events.click();
  const encoded=new TextEncoder().encode('event: branch_status\ndata: {"branch":"baseline","status":"generating"}\n\nevent: token\ndata: {"branch":"baseline","text":"A partial answer"}\n\nevent: error\ndata: {"message":"Connection interrupted"}\n\n');
  app.setGeneration({ok:true,body:new ReadableStream({start(controller){controller.enqueue(encoded);controller.close();}})});
  await app.nodes.get('run-button').events.click();
  assert.equal(app.nodes.get('baseline-output').textContent,'A partial answer');
  assert.equal(app.nodes.get('baseline-metrics').textContent,'Partial response · interrupted');
  assert.equal(app.nodes.get('hallucination-output').textContent,'No response received.');
  assert.equal(app.nodes.get('run-progress').textContent,'0 of 4 responses completed.');
  assert.equal(app.nodes.get('run-state').textContent,'GENERATION FAILED');
  assert.equal(app.nodes.get('run-button').disabled,false);
});
