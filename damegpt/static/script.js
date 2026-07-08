const messageList = document.getElementById('message-list');
const chatForm = document.getElementById('chat-form');
const promptInput = document.getElementById('prompt-input');

function appendMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  const label = document.createElement('span');
  label.className = 'role';
  label.textContent = role === 'user' ? 'You' : 'DameGPT';

  const content = document.createElement('div');
  content.textContent = text;

  wrapper.appendChild(label);
  wrapper.appendChild(content);
  messageList.appendChild(wrapper);
  messageList.scrollTop = messageList.scrollHeight;
}

async function sendPrompt(prompt) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ prompt }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Unable to fetch response');
  }

  const payload = await response.json();
  return payload.response;
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  appendMessage('user', prompt);
  promptInput.value = '';
  promptInput.focus();

  const placeholder = document.createElement('div');
  placeholder.className = 'message assistant';
  placeholder.textContent = 'DameGPT is thinking...';
  messageList.appendChild(placeholder);
  messageList.scrollTop = messageList.scrollHeight;

  try {
    const answer = await sendPrompt(prompt);
    placeholder.textContent = answer;
  } catch (error) {
    placeholder.textContent = `Error: ${error.message}`;
  }
});
