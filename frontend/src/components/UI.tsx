import {ReactNode} from 'react';
export const Card=({children,className='' }:{children:ReactNode,className?:string})=><section className={`panel ${className}`}>{children}</section>;
export const Badge=({children}:{children:ReactNode})=><span className="status-tag">{children}</span>;
export const title=(x:string)=>x.charAt(0).toUpperCase()+x.slice(1);
