import type {DocumentData, FirestoreError, Timestamp} from 'firebase/firestore';

export interface UseFireStoreReturn<T> {
	data: T;
	loading: boolean;
	error: FirestoreError | null;
}

export interface Batch extends DocumentData {
	created_at: Timestamp;
	count: number;
	prompt: string;
	transcription: string;
	user_transcriptions: string[];
}

export interface Comment extends DocumentData {
	comment: string;
	created_at: Timestamp;
}
