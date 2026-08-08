export function cropTimes(values, start, end) {
  return (values || [])
    .filter((value) => value >= start && value < end)
    .map((value) => value - start);
}

export function cropTimesWithValues(times, values, start, end) {
  const croppedTimes = [];
  const croppedValues = [];
  for (let index = 0; index < (times || []).length; index++) {
    const value = times[index];
    if (value < start || value >= end) continue;
    croppedTimes.push(value - start);
    if (index < (values || []).length) croppedValues.push(values[index]);
  }
  return [croppedTimes, croppedValues];
}

export function sourceTimes(values, sourceStart) {
  return (values || []).map((value) => value + sourceStart);
}

export function sourceTimeAtLocalFrame(frame, cropStart, fps) {
  return cropStart + frame / fps;
}

export function sourceTimeToLocalFrame(sourceTime, cropStart, fps) {
  return Math.round((sourceTime - cropStart) * fps);
}

export function waveformBinRange(
  firstFrame,
  lastFrame,
  cropStart,
  fps,
  previewStart,
  previewDuration,
  binCount,
) {
  const firstSeconds = sourceTimeAtLocalFrame(firstFrame, cropStart, fps) - previewStart;
  const lastSeconds = sourceTimeAtLocalFrame(lastFrame, cropStart, fps) - previewStart;
  const firstBin = Math.max(
    0,
    Math.min(binCount - 1, Math.floor(firstSeconds / previewDuration * binCount)),
  );
  const lastBin = Math.max(
    firstBin + 1,
    Math.min(binCount, Math.ceil(lastSeconds / previewDuration * binCount)),
  );
  return [firstBin, lastBin];
}
