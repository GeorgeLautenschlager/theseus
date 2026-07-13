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
