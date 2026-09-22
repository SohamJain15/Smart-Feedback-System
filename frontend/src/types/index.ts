export type Feedback={id:number;text:string;category:string;rating:number;predicted_sentiment:string;predicted_topic:string;predicted_priority:string;sentiment_confidence:number;topic_confidence:number;priority_confidence:number;actual_sentiment?:string;actual_topic?:string;actual_priority?:string;reviewed:boolean;created_at:string};
export type Prediction={label:string;confidence:number};
