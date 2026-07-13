const debugLog = document.getElementById('debug-log');
let pendingOlderScroll = null;
let stickToBottom = true;

function isNearBottom(el, thresholdPx = 40) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= thresholdPx;
}

if (debugLog) {
  debugLog.scrollTop = debugLog.scrollHeight;
  debugLog.addEventListener('scroll', () => {
    stickToBottom = isNearBottom(debugLog);
  });
}

// Reverse infinite scroll (loading older events above the current view)
// shifts scroll position unless compensated: capture scrollHeight/scrollTop
// right before the /debug/older request, then restore the delta after the
// swap so the row the user was looking at stays under the same pixel.
document.body.addEventListener('htmx:beforeRequest', (event) => {
  const path = event.detail && event.detail.requestConfig && event.detail.requestConfig.path;
  if (debugLog && path && path.startsWith('/debug/older')) {
    pendingOlderScroll = { scrollHeight: debugLog.scrollHeight, scrollTop: debugLog.scrollTop };
  }
});

document.body.addEventListener('htmx:afterSwap', (event) => {
  const path = event.detail && event.detail.requestConfig && event.detail.requestConfig.path;
  if (debugLog && pendingOlderScroll && path && path.startsWith('/debug/older')) {
    const delta = debugLog.scrollHeight - pendingOlderScroll.scrollHeight;
    debugLog.scrollTop = pendingOlderScroll.scrollTop + delta;
    pendingOlderScroll = null;
  }
});

// Live events arrive as OOB appends (htmx:oobAfterSwap, not htmx:afterSwap) —
// only auto-follow them if the user was already scrolled to the bottom.
document.body.addEventListener('htmx:oobAfterSwap', () => {
  if (debugLog && stickToBottom) debugLog.scrollTop = debugLog.scrollHeight;
});
