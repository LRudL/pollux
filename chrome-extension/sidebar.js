/**
 * Sidebar script for DunClone
 * Handles sidebar interactions and content updates
 */

// DOM elements
const analysisEl = document.getElementById('analysis-content');
const closeButton = document.querySelector('.close-button');
const chatMessagesEl = document.getElementById('chat-messages');
const chatInputField = document.getElementById('chat-input-field');
const sendMessageBtn = document.getElementById('send-message-btn');

// Chat message storage
let chatMessages = [];

// Listen for messages from the parent page
window.addEventListener('message', function(event) {
  // Verify the sender is from our extension
  if (event.data && event.data.type === 'ECON_SIDEBAR_DATA') {
    updateSidebarContent(event.data);
  }
});

/**
 * Updates the sidebar content with data
 * @param {Object} data - The data to display
 */
function updateSidebarContent(data) {
  if (data.analysis) {
    // Format the analysis content with proper HTML structure
    let formattedAnalysis = formatAnalysisContent(data.analysis);
    analysisEl.innerHTML = formattedAnalysis;
    analysisEl.classList.remove('loading');
  } else if (data.error) {
    analysisEl.innerHTML = `<div class="error-message"><i class="fas fa-exclamation-circle"></i> Error: ${data.error}</div>`;
    analysisEl.classList.remove('loading');
    analysisEl.classList.add('error');
  }

  if (data.chatMessages) {
    chatMessages = data.chatMessages;
    renderChatMessages();
  }
}

/**
 * Formats the analysis content with appropriate HTML structure
 * @param {string} analysisText - The raw analysis text from the LLM
 * @returns {string} - Formatted HTML content
 */
function formatAnalysisContent(analysisText) {
  // Replace numbered list items like "1." with proper HTML
  let formatted = analysisText.replace(/(\d+\.)\s+\*\*([^*]+)\*\*/g, '<h3>$1 $2</h3>');
  
  // Replace other bold elements
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  
  // Replace italic elements
  formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  
  // Replace paragraphs (double newlines)
  formatted = formatted.replace(/\n\n/g, '</p><p>');
  
  // Replace single newlines
  formatted = formatted.replace(/\n/g, '<br>');
  
  // Wrap the content in paragraph tags
  formatted = '<p>' + formatted + '</p>';
  
  // Fix any empty paragraphs
  formatted = formatted.replace(/<p>\s*<\/p>/g, '');
  
  return formatted;
}

/**
 * Sends a chat message to the background script
 */
function sendChatMessage() {
  const message = chatInputField.value.trim();
  
  if (!message) return;
  
  // Clear the input field
  chatInputField.value = '';
  
  // Add user message to the chat
  addMessageToChat('user', message);
  
  // Send message to parent page to forward to background script
  window.parent.postMessage({
    type: 'SEND_CHAT_MESSAGE',
    message
  }, '*');
}

/**
 * Adds a message to the chat UI
 * @param {string} role - 'user' or 'assistant'
 * @param {string} content - The message content
 */
function addMessageToChat(role, content) {
  // Create message element
  const messageEl = document.createElement('div');
  messageEl.className = `message ${role}-message`;
  
  // Format text content (handle simple markdown)
  let formattedContent = content
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
  
  messageEl.innerHTML = formattedContent;
  
  // Add a timestamp div
  const timestamp = document.createElement('div');
  timestamp.className = 'message-timestamp';
  timestamp.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  messageEl.appendChild(timestamp);
  
  // Add to UI
  chatMessagesEl.appendChild(messageEl);
  
  // Add a clearfix to properly handle floated elements
  const clearfix = document.createElement('div');
  clearfix.style.clear = 'both';
  chatMessagesEl.appendChild(clearfix);
  
  // Scroll to bottom of chat AND content area
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  
  // Also scroll the content area to show the new message
  const contentArea = document.querySelector('.content-area');
  if (contentArea) {
    contentArea.scrollTop = contentArea.scrollHeight;
  }
  
  // Save message to chat history
  chatMessages.push({ role, content });
}

/**
 * Renders all chat messages from the chat history
 */
function renderChatMessages() {
  // Clear existing messages
  chatMessagesEl.innerHTML = '';
  
  // Add each message
  chatMessages.forEach(msg => {
    // Create message element
    const messageEl = document.createElement('div');
    messageEl.className = `message ${msg.role}-message`;
    
    // Format text content (handle simple markdown)
    let formattedContent = msg.content
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
    
    messageEl.innerHTML = formattedContent;
    
    // Add a timestamp div
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    timestamp.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    messageEl.appendChild(timestamp);
    
    chatMessagesEl.appendChild(messageEl);
    
    // Add a clearfix to properly handle floated elements
    const clearfix = document.createElement('div');
    clearfix.style.clear = 'both';
    chatMessagesEl.appendChild(clearfix);
  });
  
  // Scroll to bottom of chat AND content area
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  
  // Also scroll the content area to show the latest messages
  const contentArea = document.querySelector('.content-area');
  if (contentArea) {
    contentArea.scrollTop = contentArea.scrollHeight;
  }
}

// Close button handler
closeButton.addEventListener('click', function() {
  // Send message to parent to close sidebar
  window.parent.postMessage({ type: 'CLOSE_ECON_SIDEBAR' }, '*');
});

// Send message when button is clicked
sendMessageBtn.addEventListener('click', () => {
  sendChatMessage();
});

// Send message when Enter is pressed (without Shift)
chatInputField.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
}); 