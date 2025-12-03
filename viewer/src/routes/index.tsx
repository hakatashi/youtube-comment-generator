import {createEffect, createSignal, For, type Component} from 'solid-js';
import {AllComments} from '~/lib/firebase';
import {useFirestore} from 'solid-firebase';
import {limit, orderBy, query} from 'firebase/firestore';
import type {Comment} from '~/lib/schema';
import {Set, List} from 'immutable';

import styles from './index.module.css';

const DEFAULT_COMMENT_INTERVAL = 30_000; // 30 seconds
const MAX_COMMENTS = 50;

const randint = (max: number) => {
	return Math.floor(Math.random() * max);
};

const Index: Component = () => {
	const commentSources = useFirestore(query(
		AllComments,
		orderBy('created_at', 'desc'),
		limit(50),
	));

	const [comments, setComments] = createSignal<List<Comment>>(List());
	const [commentsPool, setCommentsPool] = createSignal<List<Comment>>(List());
	const [processedComments, setProcessedComments] = createSignal<Set<string>>(Set());
	const [commentUpdateSchedule, setCommentUpdateSchedule] = createSignal<List<number>>(List());

	let commentsRef!: HTMLDivElement;

	createEffect(() => {
		console.log('Comments updated:', commentSources.data);
		if (commentSources.error) {
			console.error('Error fetching comments:', commentSources.error);
		}

		if (commentSources.data) {
			for (const comment of commentSources.data) {
				setProcessedComments((prev) => {
					if (!prev.has(comment.id)) {
						setCommentsPool((prev) => prev.push(comment));
						return prev.add(comment.id);
					}
					return prev;
				});
			}
		}
	});

	createEffect<number>((oldCommentsPoolSize) => {
		const now = Date.now();
		const newCommentsPoolSize = commentsPool().size;

		if (
			oldCommentsPoolSize === undefined ||
			newCommentsPoolSize > oldCommentsPoolSize
		) {
			const newSchedules: number[] = [];
			for (const _i of Array(newCommentsPoolSize).keys()) {
				newSchedules.push(now + randint(DEFAULT_COMMENT_INTERVAL));
			}
			newSchedules.sort((a, b) => a - b);
			setCommentUpdateSchedule(List(newSchedules));
			console.log('Updated comment update schedule:', newSchedules);
		}

		return newCommentsPoolSize;
	});

	const intervalId = setInterval(() => {
		while (true) {
			const schedule = commentUpdateSchedule().first();
			if (schedule === undefined || schedule > Date.now()) {
				break;
			}

			// Time to update a comment
			setCommentUpdateSchedule((prev) => prev.shift());
			const nextComment = commentsPool().first();
			if (nextComment) {
				setCommentsPool((prev) => prev.shift());
				setComments((prev) => {
					let newComments = prev.push(nextComment);
					if (newComments.size > MAX_COMMENTS) {
						newComments = newComments.shift();
					}
					return newComments;
				});
				commentsRef.scrollTop = commentsRef.scrollHeight;
			}
		}
	}, 1000 / 30);

	createEffect(() => {
		return () => clearInterval(intervalId);
	});

	return (
		<div class={styles.container}>
			<div class={styles.comments} ref={commentsRef}>
				<For each={comments().toArray()}>
					{(commentData: Comment) => (
						<div class={styles.comment}>
							{commentData.comment}
						</div>
					)}
				</For>
			</div>
		</div>
	);
};

export default Index;
