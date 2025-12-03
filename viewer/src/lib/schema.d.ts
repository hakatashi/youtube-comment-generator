import type {DocumentData, FirestoreError, Timestamp} from 'firebase/firestore';

export interface UseFireStoreReturn<T> {
	data: T;
	loading: boolean;
	error: FirestoreError | null;
}

export interface Comment extends DocumentData {
	comment: string;
	prompt: string;
	transcription: string;
	user_transcriptions: string[];
	created_at: Timestamp;
}
