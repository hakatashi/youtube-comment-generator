import type {Timestamp} from 'firebase/firestore';

export interface Comment {
	comment: string;
	prompt: string;
	transcription: string;
	user_transcriptions: string[];
	created_at: Timestamp;
}
