import {initializeApp} from 'firebase/app';
import {
	getFirestore,
	collection,
	type CollectionReference,
} from 'firebase/firestore';
import type {Comment} from './schema';

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

const Comments = collection(db, 'comments') as CollectionReference<Comment>;

export {app as default, db, Comments};
