import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {defineConfig} from '@solidjs/start/config';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
	vite: {
		plugins: [],
		build: {
			target: 'esnext',
		},
		optimizeDeps: {
			exclude: [
				'firebase/firestore',
				'@firebase/firestore',
			],
		},
		resolve: {
			alias: {
				'@firebase/app': path.resolve(
					__dirname,
					'node_modules/@firebase/app/dist/esm/index.esm.js',
				),
			},
		},
	},
	ssr: false,
	server: {
		compatibilityDate: '2024-11-07',
		esbuild: {
			options: {
				supported: {
					'top-level-await': true,
				},
			},
		},
	},
});
