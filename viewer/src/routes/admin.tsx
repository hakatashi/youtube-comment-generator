import {createEffect, createSignal, For, Show, type Component} from 'solid-js';
import {Batches, storage} from '~/lib/firebase';
import {useFirestore} from 'solid-firebase';
import {limit, orderBy, query, collection, getDocs, CollectionReference} from 'firebase/firestore';
import {ref, getDownloadURL} from 'firebase/storage';
import type {Batch, Comment} from '~/lib/schema';
import {db} from '~/lib/firebase';

import styles from './admin.module.css';

const Admin: Component = () => {
	const batchesQuery = useFirestore(query(
		Batches,
		orderBy('created_at', 'desc'),
		limit(20),
	));

	const [batchesWithComments, setBatchesWithComments] = createSignal<Array<{batch: Batch & {id: string}, comments: Comment[], imageUrls?: string[]}>>([]);
	const [expandedPrompts, setExpandedPrompts] = createSignal<Set<string>>(new Set());
	const [expandedImages, setExpandedImages] = createSignal<Set<string>>(new Set());

	createEffect(async () => {
		if (batchesQuery.data) {
			const batchData = await Promise.all(
				batchesQuery.data.map(async (batch) => {
					const commentsRef = collection(db, 'batches', batch.id, 'comments') as CollectionReference<Comment>;
					const commentsSnapshot = await getDocs(query(commentsRef, orderBy('index', 'asc')));
					const comments = commentsSnapshot.docs.map((doc) => ({
						id: doc.id,
						...doc.data(),
					} as Comment));

					// 画像URLを取得
					let imageUrls: string[] | undefined;
					if (batch.image_paths && batch.image_paths.length > 0) {
						try {
							imageUrls = await Promise.all(
								batch.image_paths.map(async (path) => {
									const imageRef = ref(storage, path);
									return await getDownloadURL(imageRef);
								})
							);
						} catch (error) {
							console.error('Failed to load image URLs:', error);
						}
					}

					return {
						batch: {id: batch.id, ...batch},
						comments,
						imageUrls,
					};
				})
			);

			setBatchesWithComments(batchData);
		}
	});

	const formatDate = (timestamp: any) => {
		if (!timestamp) return 'N/A';
		const date = timestamp.toDate();
		return date.toLocaleString('ja-JP', {
			year: 'numeric',
			month: '2-digit',
			day: '2-digit',
			hour: '2-digit',
			minute: '2-digit',
			second: '2-digit',
		});
	};

	const formatDuration = (seconds?: number) => {
		if (seconds === undefined || seconds === null) return 'N/A';
		return `${seconds.toFixed(2)}秒`;
	};

	const togglePrompt = (batchId: string) => {
		setExpandedPrompts((prev) => {
			const newSet = new Set(prev);
			if (newSet.has(batchId)) {
				newSet.delete(batchId);
			} else {
				newSet.add(batchId);
			}
			return newSet;
		});
	};

	const toggleImages = (batchId: string) => {
		setExpandedImages((prev) => {
			const newSet = new Set(prev);
			if (newSet.has(batchId)) {
				newSet.delete(batchId);
			} else {
				newSet.add(batchId);
			}
			return newSet;
		});
	};

	return (
		<div class={styles.container}>
			<header class={styles.header}>
				<h1>管理者用コメント閲覧ページ</h1>
				<div class={styles.stats}>
					<Show when={batchesQuery.data}>
						<span>バッチ数: {batchesQuery.data?.length || 0}</span>
					</Show>
				</div>
			</header>

			<div class={styles.batchList}>
				<Show when={batchesQuery.loading}>
					<div class={styles.loading}>読み込み中...</div>
				</Show>

				<Show when={batchesQuery.error}>
					<div class={styles.error}>エラー: {batchesQuery.error?.message}</div>
				</Show>

				<For each={batchesWithComments()}>
					{(item) => (
						<div class={styles.batchCard}>
							<div class={styles.batchHeader}>
								<span class={styles.metaItem}>
									<strong>ID:</strong> {item.batch.id.substring(0, 8)}...
								</span>
								<span class={styles.metaItem}>
									<strong>作成:</strong> {formatDate(item.batch.created_at)}
								</span>
								<span class={styles.metaItem}>
									<strong>コメント数:</strong> {item.batch.count}
								</span>
								<span class={styles.metaItem}>
									<strong>ユーザー数:</strong> {item.batch.user_ids?.length ?? 'N/A'}
								</span>
								<span class={styles.metaItem}>
									<strong>音声:</strong> {formatDuration(item.batch.audio_duration)}
								</span>
								<span class={styles.metaItem}>
									<strong>STT:</strong> {formatDuration(item.batch.stt_duration)}
								</span>
								<span class={styles.metaItem}>
									<strong>コメント生成:</strong> {formatDuration(item.batch.comment_gen_duration)}
								</span>
								<span class={styles.metaItem}>
									<strong>合計:</strong> {formatDuration(item.batch.total_duration)}
								</span>
								<Show when={item.imageUrls && item.imageUrls.length > 0}>
									<span class={styles.metaItem}>
										<strong>画像:</strong> {item.imageUrls!.length}枚
									</span>
								</Show>
								<button
									class={styles.promptToggle}
									onClick={() => togglePrompt(item.batch.id)}
								>
									{expandedPrompts().has(item.batch.id) ? 'プロンプトを隠す' : 'プロンプトを表示'}
								</button>
								<Show when={item.imageUrls && item.imageUrls.length > 0}>
									<button
										class={styles.promptToggle}
										onClick={() => toggleImages(item.batch.id)}
									>
										{expandedImages().has(item.batch.id) ? '画像を隠す' : '画像を表示'}
									</button>
								</Show>
							</div>

							<div class={styles.batchContent}>
								<Show when={expandedPrompts().has(item.batch.id)}>
									<div class={styles.section}>
										<p class={styles.prompt}>{item.batch.prompt}</p>
									</div>
								</Show>

								<Show when={expandedImages().has(item.batch.id) && item.imageUrls}>
									<div class={styles.section}>
										<h3>使用した画像</h3>
										<div class={styles.imageGallery}>
											<For each={item.imageUrls}>
												{(imageUrl, index) => (
													<div class={styles.imageContainer}>
														<img
															src={imageUrl}
															alt={`Screenshot ${index() + 1}`}
															class={styles.screenshotImage}
														/>
														<Show when={item.batch.image_paths && item.batch.image_paths[index()]}>
															<p class={styles.imagePath}>{item.batch.image_paths![index()]}</p>
														</Show>
													</div>
												)}
											</For>
										</div>
									</div>
								</Show>

								<Show when={item.batch.transcription}>
									<div class={styles.section}>
										<h3>文字起こし</h3>
										<p class={styles.transcription}>{item.batch.transcription}</p>
									</div>
								</Show>

								<div class={styles.commentList}>
									<For each={item.comments}>
										{(comment) => (
											<div class={styles.commentItem}>
												{comment.comment}
											</div>
										)}
									</For>
									<Show when={item.comments.length === 0}>
										<div class={styles.noComments}>コメントがありません</div>
									</Show>
								</div>
							</div>
						</div>
					)}
				</For>
			</div>
		</div>
	);
};

export default Admin;
