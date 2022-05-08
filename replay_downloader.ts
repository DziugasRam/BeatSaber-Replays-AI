import * as fs from "fs";
import fetch from "node-fetch";

const formatNotSupportedErrorMessage = "Format not supported";

const getReplay = async (playerId: string, leaderboardId: string): Promise<string> => {
	return await new Promise(async (res, rej) => {
		setTimeout(() => {
			rej();
		}, 60 * 1000);
		try {
			const url = `https://sspreviewdecode.azurewebsites.net/?playerID=${playerId}&songID=${leaderboardId}`;
			const response = await fetch(url);
			const responseText = await response.text();
			if (responseText === `"{\\"errorMessage\\": \\"Old replay format not supported.\\"}"`) {
				rej(formatNotSupportedErrorMessage);
				return;
			}
			res(responseText);
		} catch (err: any) {
			rej(err);
		}
	});
};

const getReplayFileName = (playerId: string, leaderboardId: string): string => {
	return `${playerId}-${leaderboardId}.txt`;
};

const getReplayDir = (playerId: string, leaderboardId: string): string => {
	return `replays/${leaderboardId}/`;
};

const saveReplayToFile = (replay: string, playerId: string, leaderboardId: string): void => {
	const fileName = getReplayFileName(playerId, leaderboardId);
	const replayDir = getReplayDir(playerId, leaderboardId);
	fs.mkdirSync(replayDir, { recursive: true });
	fs.writeFileSync(`${replayDir}${fileName}`, replay);
};

interface ScoresaberLeaderboardsResponse {
	leaderboards: ScoresaberLeaderboard[];
	metadata: unknown;
}

interface ScoresaberLeaderboard {
	id: string;
	starRating: number;
	songHash: string;
}

const getScoresaberLeaderboards = async (page: number): Promise<ScoresaberLeaderboard[]> => {
	// Ranked leaderboards sorted by most stars
	const url = `https://scoresaber.com/api/leaderboards?category=3&maxStar=50&minStar=0&page=${page}&ranked=1&sort=0&verified=1`;
	const response = await fetch(url);
	const responseJson = (await response.json()) as ScoresaberLeaderboardsResponse;
	return responseJson.leaderboards;
};

interface ScoresaberLeaderboardResponse {
	scores: ScoresaberLeaderboardScore[];
	metadata: unknown;
}

interface ScoresaberLeaderboardScore {
	leaderboardPlayerInfo: {
		id: string;
	};
	hasReplay: boolean;
	modifiers: string;
}

const getScoresaberLeaderboard = async (
	leaderboardId: string,
	page: number,
	retryCount = 0
): Promise<ScoresaberLeaderboardScore[]> => {
	try {
		const url = `https://scoresaber.com/api/leaderboard/by-id/${leaderboardId}/scores?page=${page}`;
		const response = await fetch(url);
		const responseJson = (await response.json()) as ScoresaberLeaderboardResponse;
		return responseJson.scores;
	} catch (err: any) {
		console.log(`Failed: ${leaderboardId} - ${page} - ${retryCount}`);
		if (retryCount > 20) {
			console.log("Failed, LLLLLL");
			return [];
		}
		await new Promise((res, rej) => {
			setTimeout(() => {
				res(undefined);
			}, 5000);
		});
		return await getScoresaberLeaderboard(leaderboardId, page, retryCount + 1);
	}
};

const getScoresaberLeaderboardsPages = async (startPage: number, endPage: number): Promise<ScoresaberLeaderboard[]> => {
	const results: ScoresaberLeaderboard[] = [];

	for (let page = startPage; page <= endPage; ++page) {
		const leaderboards = await getScoresaberLeaderboards(page);
		results.push(...leaderboards);
	}

	return results;
};

const getFilteredScoresaberLeaderboardScores = async (
	leaderboardId: string,
	maxPages: number,
	filterPredicate: (score: ScoresaberLeaderboardScore) => boolean,
	stopOnNoResults: boolean = true
): Promise<ScoresaberLeaderboardScore[]> => {
	const results: ScoresaberLeaderboardScore[] = [];

	for (let page = 1; page <= maxPages; ++page) {
		const leaderboardScores = await getScoresaberLeaderboard(leaderboardId, page);
		const filteredLeaderboardScores = leaderboardScores.filter(filterPredicate);
		if (stopOnNoResults && filteredLeaderboardScores.length == 0) break;
		results.push(...filteredLeaderboardScores);
	}

	return results;
};

const getAllLeaderboardsWithScores = async (
	startLeaderboardsPage: number,
	endLeaderboardsPage: number,
	maxPagesPerLeaderboard: number
): Promise<[ScoresaberLeaderboard, ScoresaberLeaderboardScore[]][]> => {
	const leaderboards = await getScoresaberLeaderboardsPages(startLeaderboardsPage, endLeaderboardsPage);
	const results: [ScoresaberLeaderboard, ScoresaberLeaderboardScore[]][] = [];

	for (const leaderboard of leaderboards) {
		const filterPredicate = (score: ScoresaberLeaderboardScore) => score.hasReplay && score.modifiers.length === 0;
		const leaderboardScores = await getFilteredScoresaberLeaderboardScores(
			leaderboard.id,
			maxPagesPerLeaderboard,
			filterPredicate,
			true
		);
		results.push([leaderboard, leaderboardScores]);
	}

	return results;
};

const getAndSaveReplay = async (leaderboardId: string, playerId: string): Promise<void> => {
	const replay = await getReplay(playerId, leaderboardId);
	if (replay.length < 10000 && replay.includes("503")) throw new Error(replay);
	if (replay.length < 10000 && replay.includes("403")) {
		console.log(`403, :skull-emoji: ${leaderboardId} - ${playerId}`);
		return;
	}
	if (replay.length < 10000) {
		console.log(`???, :skull-emoji: ${leaderboardId} - ${playerId}`);
		return;
	}
	saveReplayToFile(replay, playerId, leaderboardId);
};

const waitForRemainingRetryCountAsync = async (remainingRetries: number) => {
	const ms = (10 - remainingRetries) ** 2 * 1000;
	return await new Promise((resolve) => setTimeout(resolve, ms));
};

const getAndSaveReplayWithRetry = async (
	leaderboardId: string,
	playerId: string,
	remainingRetries: number = 10
): Promise<void> => {
	try {
		console.log(new Date().toLocaleString(), `Starting replay ${leaderboardId} ${playerId}`);
		await getAndSaveReplay(leaderboardId, playerId);
		console.log(new Date().toLocaleString(), `Saved replay ${leaderboardId} ${playerId}`);
	} catch (error: any) {
		if (JSON.stringify(error)?.includes(formatNotSupportedErrorMessage)) {
			console.log(new Date().toLocaleString(), `Format not supported ${leaderboardId} ${playerId}`);
			return;
		}

		if (remainingRetries > 0) {
			console.log("");
			console.error(error);
			await waitForRemainingRetryCountAsync(remainingRetries);
			console.log(
				new Date().toLocaleString(),
				`Retrying for ${leaderboardId} ${playerId}. Retries left - ${remainingRetries - 1}`
			);
			await getAndSaveReplayWithRetry(leaderboardId, playerId, remainingRetries - 1);
		} else {
			console.log("");
			console.log("-------------------------------------------------");
			console.error(error);
			console.log(new Date().toLocaleString(), `Failed to load replay ${leaderboardId} ${playerId}`);
			console.log("-------------------------------------------------");
		}
	}
};

const main = async () => {
	const leaderboardsWithScores = await getAllLeaderboardsWithScores(7, 50, 5);
	console.log("Number of leaderboards:", leaderboardsWithScores.length);
	console.log("Number of scores:", leaderboardsWithScores.flatMap(([leaderboard, scores]) => scores).length);

	let doo = false;

	for (const [leaderboard, scores] of leaderboardsWithScores) {
		for (const score of scores) {
			// leaderboard id and player id of the last saved replay
			if (leaderboard.id == "295303" && score.leaderboardPlayerInfo.id == "76561197995162898") doo = true;
			if (!doo) continue;
			await getAndSaveReplayWithRetry(leaderboard.id, score.leaderboardPlayerInfo.id);
			await new Promise((res, rej) => {
				setTimeout(() => {
					res(undefined);
				}, 10000);
			});
		}
	}
};


console.log("don't use unless you need only a couple of replays. DM me if a lot of replays are needed")

// main();

// const test = async () => {
//     await getAndSaveReplayWithRetry("280301", "2085408448198355");
// }
// test();
