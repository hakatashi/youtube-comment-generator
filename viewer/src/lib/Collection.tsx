import type {DocumentData} from 'firebase/firestore';
import {For, type JSX, Match, Switch} from 'solid-js';
import type {UseFireStoreReturn} from './schema.ts';

interface Props<T extends DocumentData> {
	data: UseFireStoreReturn<T[] | null | undefined> | null | undefined;
	children: (item: T, index: () => number) => JSX.Element;
	empty?: JSX.Element;
}

const Skeleton = (_props: {variant: 'text'}) => <span>Loading...</span>;

const Collection = <T extends DocumentData>(props: Props<T>) => (
	<Switch>
		<Match when={props.data === null}>
			<Skeleton variant="text" />
		</Match>
		<Match when={props.data?.loading}>
			<Skeleton variant="text" />
		</Match>
		<Match when={props.data?.error}>
			<span class="load-error">{props.data?.error?.toString()}</span>
		</Match>
		<Match when={props.data?.data} keyed={true}>
			{(docs) => (
				<Switch>
					<Match when={docs.length === 0 && props.empty !== undefined}>
						{props.empty}
					</Match>
					<Match when={true}>
						<For each={docs}>{(doc, index) => props.children(doc, index)}</For>
					</Match>
				</Switch>
			)}
		</Match>
		<Match when={true}>
			<span class="load-error">Load error occurred.</span>
		</Match>
	</Switch>
);

export default Collection;
