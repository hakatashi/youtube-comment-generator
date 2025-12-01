import {type Component} from 'solid-js';
import {Comments} from '~/lib/firebase';
import {useFirestore} from 'solid-firebase';
import Collection from '~/lib/Collection';
import {orderBy, query} from 'firebase/firestore';
import type {Comment} from '~/lib/schema';

import styles from './index.module.css';

const Index: Component = () => {
	const comments = useFirestore(query(Comments, orderBy('created_at', 'desc')));

	const formatDate = (timestamp: any) => {
		if (!timestamp) return '';
		const date = timestamp.toDate();
		return new Intl.DateTimeFormat('ja-JP', {
			year: 'numeric',
			month: '2-digit',
			day: '2-digit',
			hour: '2-digit',
			minute: '2-digit',
			second: '2-digit',
		}).format(date);
	};

	return (
		<div class={styles.container}>
			<h1 class={styles.title}>VTuber コメントジェネレーター</h1>
			<div class={styles.comments}>
				<Collection data={comments}>
					{(commentData: Comment) => (
						<div class={styles.commentCard}>
							<div class={styles.commentHeader}>
								<span class={styles.timestamp}>
									{formatDate(commentData.created_at)}
								</span>
							</div>
							<div class={styles.comment}>
								{commentData.comment}
							</div>
							<details class={styles.metadata}>
								<summary>詳細</summary>
								<div class={styles.metadataContent}>
									<div class={styles.metadataSection}>
										<h3>文字起こし</h3>
										<p>{commentData.transcription}</p>
									</div>
									{commentData.user_transcriptions && commentData.user_transcriptions.length > 0 && (
										<div class={styles.metadataSection}>
											<h3>ユーザーごとの発言</h3>
											<ul>
												{commentData.user_transcriptions.map((ut: string) => (
													<li>{ut}</li>
												))}
											</ul>
										</div>
									)}
								</div>
							</details>
						</div>
					)}
				</Collection>
			</div>
		</div>
	);
};

export default Index;
