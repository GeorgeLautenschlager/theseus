function autoGrow(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function updateSendButton(textarea) {
  const button = document.getElementById('composer-send');
  if (!button) return;
  button.disabled = textarea.disabled || textarea.value.trim().length === 0;
}

function scrollMessagesToBottom() {
  const messages = document.getElementById('messages');
  if (messages) messages.scrollTop = messages.scrollHeight;
}

// Composer controls are replaced wholesale (hx-swap-oob) whenever the agent
// starts/finishes a reply, so listeners are delegated on document rather
// than bound to the (possibly stale) textarea/button elements directly.
document.addEventListener('input', (event) => {
  if (event.target && event.target.id === 'composer-input') {
    autoGrow(event.target);
    updateSendButton(event.target);
  }
});

document.addEventListener('keydown', (event) => {
  if (event.target && event.target.id === 'composer-input' && event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    const form = event.target.closest('form');
    const button = document.getElementById('composer-send');
    if (form && button && !button.disabled) {
      form.requestSubmit();
    }
  }
});

document.body.addEventListener('htmx:afterSwap', scrollMessagesToBottom);
document.body.addEventListener('htmx:oobAfterSwap', scrollMessagesToBottom);

// -- notifications --------------------------------------------------------
//
// The server marks a finished reply by swapping in #chat-notify carrying the
// message id and a plain-text preview (see _assistant_reply_fragment.html).
// While the page is unfocused that becomes an unread count in the title and,
// when permitted, a desktop notification. Notification is only defined in a
// secure context, so http://<lan-ip>:<port> gets the title count alone.

const NOTIFY_PREF_KEY = 'theseus.notifications';
const BASE_TITLE = document.title;
const seenNotifyIds = new Set();
let unreadCount = 0;

function notificationsSupported() {
  return 'Notification' in window;
}

function notifyPrefEnabled() {
  try {
    return localStorage.getItem(NOTIFY_PREF_KEY) !== 'off';
  } catch (error) {
    return true; // storage blocked (private mode, cookies off) — permission still governs
  }
}

function setNotifyPref(value) {
  try {
    localStorage.setItem(NOTIFY_PREF_KEY, value);
  } catch (error) {
    /* nothing to persist to; the current page still honours the click */
  }
}

function renderBellState() {
  const button = document.getElementById('notify-toggle');
  if (!button) return;
  button.classList.remove('is-on', 'is-off', 'is-denied', 'is-unavailable');

  if (!notificationsSupported()) {
    button.classList.add('is-unavailable');
    button.title = 'Desktop notifications need localhost or HTTPS';
    button.setAttribute('aria-pressed', 'false');
    return;
  }

  if (Notification.permission === 'denied') {
    button.classList.add('is-denied');
    button.title = 'Notifications are blocked — unblock this site in your browser settings';
    button.setAttribute('aria-pressed', 'false');
    return;
  }

  const on = Notification.permission === 'granted' && notifyPrefEnabled();
  button.classList.add(on ? 'is-on' : 'is-off');
  button.setAttribute('aria-pressed', on ? 'true' : 'false');
  button.title = Notification.permission === 'granted'
    ? (on ? 'Notifications on — click to mute' : 'Notifications muted — click to unmute')
    : 'Click to enable desktop notifications';
}

function toggleNotifications() {
  if (!notificationsSupported() || Notification.permission === 'denied') return;

  if (Notification.permission === 'default') {
    // Requesting from a click satisfies the user-gesture requirement that
    // makes Chrome auto-deny bare on-load requests.
    Notification.requestPermission().then((permission) => {
      if (permission === 'granted') setNotifyPref('on');
      renderBellState();
    });
    return;
  }

  setNotifyPref(notifyPrefEnabled() ? 'off' : 'on');
  renderBellState();
}

function raiseNotification(preview, msgId) {
  if (!notificationsSupported()) return;
  if (Notification.permission !== 'granted' || !notifyPrefEnabled()) return;
  try {
    // Every unfocused tab raises this; the shared tag collapses them into one.
    const notification = new Notification('Theseus', { body: preview, tag: msgId });
    notification.onclick = () => {
      window.focus();
      notification.close();
    };
  } catch (error) {
    /* some browsers only allow notifications from a service worker */
  }
}

function handleNotifySignal() {
  const signal = document.getElementById('chat-notify');
  if (!signal) return;
  const msgId = signal.dataset.msgId;
  if (!msgId || seenNotifyIds.has(msgId)) return;
  seenNotifyIds.add(msgId);
  if (document.hasFocus()) return;

  unreadCount += 1;
  document.title = '(' + unreadCount + ') ' + BASE_TITLE;
  raiseNotification(signal.dataset.preview || 'New message', msgId);
}

function clearUnread() {
  if (unreadCount === 0) return;
  unreadCount = 0;
  document.title = BASE_TITLE;
}

document.body.addEventListener('htmx:oobAfterSwap', handleNotifySignal);
window.addEventListener('focus', clearUnread);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) clearUnread();
});

const notifyToggle = document.getElementById('notify-toggle');
if (notifyToggle) notifyToggle.addEventListener('click', toggleNotifications);
renderBellState();
