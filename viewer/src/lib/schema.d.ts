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
	audio_duration?: number;
	stt_duration?: number;
	comment_gen_duration?: number;
	total_duration?: number;
	user_ids?: number[];
}

export interface Comment extends DocumentData {
	comment: string;
	created_at: Timestamp;
	index: number;
}
