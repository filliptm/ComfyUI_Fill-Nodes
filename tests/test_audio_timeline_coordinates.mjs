import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const moduleSource = await readFile(
  new URL("../web/nodes/audio/audio_timeline_coordinates.js", import.meta.url),
  "utf8",
);
const {
  cropTimes,
  cropTimesWithValues,
  sourceTimes,
  sourceTimeAtLocalFrame,
  sourceTimeToLocalFrame,
  waveformBinRange,
} = await import(`data:text/javascript;base64,${Buffer.from(moduleSource).toString("base64")}`);

test("crop projection uses half-open bounds and preserves paired values", () => {
  assert.deepEqual(cropTimes([1, 1.5, 2], 1, 2), [0, 0.5]);
  assert.deepEqual(
    cropTimesWithValues([1, 1.5, 2], [0.9, 0.8, 0.7], 1, 2),
    [[0, 0.5], [0.9, 0.8]],
  );
});

test("waveform and analyzed beats share the same crop origin", () => {
  const fps = 24;
  const sourceTime = 10.25;
  const firstCrop = 2;
  const secondCrop = 3.5;
  const firstFrame = sourceTimeToLocalFrame(sourceTime, firstCrop, fps);
  const secondFrame = sourceTimeToLocalFrame(sourceTime, secondCrop, fps);

  assert.equal(firstFrame - secondFrame, (secondCrop - firstCrop) * fps);
  assert.equal(sourceTimeAtLocalFrame(firstFrame, firstCrop, fps), sourceTime);
  assert.equal(sourceTimeAtLocalFrame(secondFrame, secondCrop, fps), sourceTime);

  const firstBins = waveformBinRange(firstFrame, firstFrame + 1, firstCrop, fps, 0, 20, 2000);
  const secondBins = waveformBinRange(secondFrame, secondFrame + 1, secondCrop, fps, 0, 20, 2000);
  assert.deepEqual(firstBins, secondBins);
});

test("beat offset moves only the working grid", () => {
  const fps = 24;
  const cropStart = 2;
  const detectedSourceTime = 10.25;
  const detectedFrame = sourceTimeToLocalFrame(detectedSourceTime, cropStart, fps);
  const waveformSourceTime = sourceTimeAtLocalFrame(detectedFrame, cropStart, fps);
  const shiftedGridFrame = sourceTimeToLocalFrame(detectedSourceTime + 0.125, cropStart, fps);

  assert.equal(waveformSourceTime, detectedSourceTime);
  assert.equal(shiftedGridFrame - detectedFrame, 3);
});

test("legacy crop-local markers are promoted without destructive reprojection", () => {
  const source = sourceTimes([0.25, 1.25, 2.25], 4);

  assert.deepEqual(cropTimes(source, 4, 6), [0.25, 1.25]);
  assert.deepEqual(cropTimes(source, 5, 7), [0.25, 1.25]);
  assert.deepEqual(cropTimes(source, 4, 6), [0.25, 1.25]);
});
