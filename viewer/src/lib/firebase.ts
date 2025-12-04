import {initializeApp} from 'firebase/app';
import {
	getFirestore,
	collection,
	collectionGroup,
	type CollectionReference,
	type Query,
} from 'firebase/firestore';
import {getStorage} from 'firebase/storage';
import type {Batch, Comment} from './schema';

const firebaseConfig = {
	apiKey: "AIzaSyB9D6HxqiVPmXJqPRmo90_MeLlpzIZ379Y",
	authDomain: "vtuber-comment-generator.firebaseapp.com",
	projectId: "vtuber-comment-generator",
	storageBucket: "vtuber-comment-generator.firebasestorage.app",
	messagingSenderId: "809393940820",
	appId: "1:809393940820:web:58dab59b574e6e014540c1"
};

const app = initializeApp(firebaseConfig);

const db = getFirestore(app);
const storage = getStorage(app);

const Batches = collection(db, 'batches') as CollectionReference<Batch>;
const AllComments = collectionGroup(db, 'comments') as Query<Comment>;

export {app as default, db, storage, Batches, AllComments};
