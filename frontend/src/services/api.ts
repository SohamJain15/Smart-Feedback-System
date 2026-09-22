// In development Vite forwards /api to FastAPI.  Keeping this relative also
// avoids localhost IPv4/IPv6 mismatches and works when opened from another host.
const BASE='/api';
export async function api<T=any>(path:string, options?:RequestInit):Promise<T>{const r=await fetch(BASE+path,{headers:{'Content-Type':'application/json'},...options}); if(!r.ok) throw new Error((await r.json().catch(()=>({detail:r.statusText}))).detail||'Request failed'); return r.json()}
