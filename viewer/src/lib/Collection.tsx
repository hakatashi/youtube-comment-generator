import {type JSX, For, Show} from 'solid-js';
import type {FirestoreError, QuerySnapshot} from 'firebase/firestore';

export interface CollectionProps<T> {
	data: {
		loading: boolean;
		error: FirestoreError | null;
		data: QuerySnapshot<T> | null;
	};
	fallback?: JSX.Element;
	children: (data: T) => JSX.Element;
}

export default function Collection<T>(props: CollectionProps<T>) {
	return (
		<Show when={!props.data.loading} fallback={props.fallback || <div>Loading...</div>}>
			<Show when={!props.data.error} fallback={<div>Error: {props.data.error?.message}</div>}>
				<Show when={props.data.data}>
					<For each={props.data.data?.docs}>
						{(doc) => props.children(doc.data())}
					</For>
				</Show>
			</Show>
		</Show>
	);
}
