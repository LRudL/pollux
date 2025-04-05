/**
 * Content script for DunClone
 * This script scans webpages for economics-related content and injects UI elements
 */

// List of economics-related keywords to search for
const ECON_KEYWORDS = [
  'economics', 'economy', 'economic', 'GDP', 'inflation', 'market', 'markets',
  'recession', 'fiscal', 'monetary', 'policy', 'finance', 'financial', 'stock',
  'stocks', 'bond', 'bonds', 'interest rate', 'supply', 'demand', 'price',
  'prices', 'exchange rate', 'currency', 'currencies', 'trade', 'deficit',
  'surplus', 'unemployment', 'employment', 'labor', 'wage', 'wages',
  'investment', 'capital', 'consumption', 'macroeconomics', 'microeconomics',
  'equilibrium', 'elasticity', 'growth', 'banking', 'central bank', 'fed',
  'federal reserve', 'ECB', 'IMF', 'world bank'
];

// Main variables
let isEconomicsRelevant = false;
let sidebarInjected = false;
let triggerButtonInjected = false;
let chatMessages = [];
let triggerButtonPosition = { right: '20px', top: '20px' }; // Store button position for animation

// For debugging
let debugMode = true;

/**
 * Debug logging function
 */
function debug(...args) {
  if (debugMode) {
    console.log('[DunClone Debug]', ...args);
  }
}

/**
 * Force button to appear for testing purposes
 */
function forceShowButton() {
  isEconomicsRelevant = true;
  injectTriggerButton();
  debug('Button forcefully shown for debugging');
}

/**
 * Checks if the current page contains economics-related content
 * @returns {boolean} - Whether the page is economics-relevant
 */
function checkForEconomicsContent() {
  debug('Checking for economics content...');
  
  try {
    const bodyText = document.body.textContent.toLowerCase();
    
    // Always return true for debugging if needed
    if (debugMode) {
      debug('Debug mode active - forcing page to be recognized as economics-relevant');
      return true;
    }
    
    // Check for presence of economics keywords
    for (const keyword of ECON_KEYWORDS) {
      if (bodyText.includes(keyword.toLowerCase())) {
        debug(`Found keyword "${keyword}"`);
        return true;
      }
    }
    
    debug('No economics keywords found on this page');
    return false;
  } catch (error) {
    console.error('DunClone: Error checking for economics content:', error);
    return false;
  }
}

/**
 * Extracts all text content from the page
 * @returns {string} - Extracted text
 */
function extractPageContent() {
  try {
    // Get all text from the page
    const bodyText = document.body.innerText;
    return bodyText;
  } catch (error) {
    console.error('DunClone: Error extracting page content:', error);
    return 'Error extracting content from page.';
  }
}

/**
 * Ensures the icon for the sidebar is ready
 * @returns {Promise<string>} - URL of the icon
 */
async function ensureIcon() {
  try {
    // Try to get the URL of the duncan-face.jpeg icon
    const iconUrl = chrome.runtime.getURL('icons/duncan-face.jpeg');
    
    // Check if the icon is accessible by loading it
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(iconUrl);
      img.onerror = () => {
        console.warn('DunClone: Could not load duncan-face.jpeg, using fallback');
        resolve(''); // Return empty string to use text fallback
      };
      img.src = iconUrl;
    });
  } catch (error) {
    console.error('DunClone: Error ensuring icon:', error);
    return ''; // Return empty string to use text fallback
  }
}

/**
 * Injects the floating trigger button into the page
 */
function injectTriggerButton() {
  try {
    if (triggerButtonInjected) {
      debug('Trigger button already injected, skipping');
      return;
    }
    
    debug('Injecting trigger button');
    
    const triggerButton = document.createElement('button');
    triggerButton.id = 'econ-trigger';
    triggerButton.textContent = 'D'; // Using simple text for reliability
    triggerButton.title = 'Ask DunClone';
    
    // Add click event with debug logging and error handling
    triggerButton.onclick = function(e) {
      debug('🔴 Trigger button clicked via onclick');
      
      try {
        // Store button position for animation origin
        const buttonRect = triggerButton.getBoundingClientRect();
        triggerButtonPosition = {
          right: window.innerWidth - buttonRect.right + 'px',
          top: buttonRect.top + 'px'
        };
        
        debug('Button position:', triggerButtonPosition);
        
        if (!sidebarInjected) {
          debug('Sidebar not yet injected, calling injectSidebar()');
          injectSidebar();
        } else {
          debug('Sidebar already injected, toggling visibility');
          toggleSidebar();
        }
      } catch (error) {
        console.error('DunClone: Error in button click handler:', error);
        alert('Error opening DunClone sidebar: ' + error.message);
      }
      
      // Stop event propagation
      e.preventDefault();
      e.stopPropagation();
      return false;
    };
    
    // Append to body and mark as injected
    document.body.appendChild(triggerButton);
    triggerButtonInjected = true;
    debug('Trigger button injected successfully');
    
    // Add a test click handler to document for debugging
    if (debugMode) {
      document.addEventListener('click', () => {
        debug('Document click detected - checking if propagation is properly stopped');
      });
    }
  } catch (error) {
    console.error('DunClone: Error injecting trigger button:', error);
  }
}

/**
 * Creates and injects the sidebar into the page
 */
async function injectSidebar() {
  try {
    if (sidebarInjected) {
      debug('Sidebar already injected, skipping');
      return;
    }
    
    debug('Injecting sidebar');
    
    // Get the icon URL, waiting for it to be ready
    const iconUrl = await ensureIcon();
    const iconHtml = iconUrl ? 
      `<img src="${iconUrl}" alt="DunClone">` : 
      '<div class="icon-placeholder">D</div>';
    
    // Create sidebar element
    const sidebar = document.createElement('div');
    sidebar.id = 'econ-sidebar';
    
    // Set initial position to match the trigger button
    sidebar.style.top = triggerButtonPosition.top;
    sidebar.style.right = triggerButtonPosition.right;
    sidebar.style.width = '44px';
    sidebar.style.height = '44px';
    sidebar.style.borderRadius = '50%';
    sidebar.style.overflow = 'hidden';
    sidebar.style.backgroundColor = '#0072cf'; // Match button color during animation
    sidebar.style.zIndex = '10000'; // Ensure high z-index
    
    // Create sidebar content with simple HTML structure
    sidebar.innerHTML = `
      <div class="sidebar-header">
        <div class="sidebar-title">
          ${iconHtml}
          <h2>DunClone</h2>
        </div>
      </div>
      <button class="close-button">✕</button>
      <div class="sidebar-content">
        <div class="content-card">
          <div class="card-body">
            <div class="content-area">
              <div id="analysis-content" class="loading">Analyzing economics content</div>
              <div class="chat-messages" id="chat-messages"></div>
            </div>
            <div class="chat-header">
              <span class="icon">💬</span>
              <h3>Chat with DunClone</h3>
            </div>
            <div class="chat-input">
              <textarea id="chat-input-field" placeholder="Ask me about this economic content..."></textarea>
              <button id="send-message-btn">📤</button>
            </div>
          </div>
        </div>
      </div>
    `;
    
    // Ensure the sidebar is properly injected before continuing
    document.body.appendChild(sidebar);
    sidebarInjected = true;
    debug('Sidebar element added to DOM');
    
    // Add event listener for close button
    const closeBtn = sidebar.querySelector('.close-button');
    if (closeBtn) {
      closeBtn.onclick = function(e) {
        debug('Close button clicked via onclick');
        collapseToButton();
        e.preventDefault();
        e.stopPropagation();
        return false;
      };
    } else {
      console.error('DunClone: Close button not found in sidebar');
    }
    
    // Add event listeners for chat
    const chatInputField = document.getElementById('chat-input-field');
    const sendMessageBtn = document.getElementById('send-message-btn');
    
    if (sendMessageBtn) {
      // Send message when button is clicked
      sendMessageBtn.onclick = function(e) {
        sendChatMessage();
        e.preventDefault();
        e.stopPropagation();
        return false;
      };
    } else {
      console.error('DunClone: Send message button not found');
    }
    
    if (chatInputField) {
      // Send message when Enter is pressed (without Shift)
      chatInputField.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendChatMessage();
        }
      });
    } else {
      console.error('DunClone: Chat input field not found');
    }
    
    // Show the sidebar with animation after a slight delay
    setTimeout(() => {
      debug('Calling expandFromButton to animate sidebar');
      expandFromButton(sidebar);
    }, 100);
    
    // Extract page content and request analysis
    const pageContent = extractPageContent();
    
    // Get research objective (we still need it for the analysis even though we don't show it)
    chrome.storage.local.get(['researchObjective'], function(result) {
      const researchObjective = result.researchObjective || 'Analyze this economics content';
      
      // Request analysis of the entire page content
      requestAnalysis(researchObjective, pageContent);
    });
    
    debug('Sidebar injected successfully');
  } catch (error) {
    console.error('DunClone: Error injecting sidebar:', error);
    alert('Error creating DunClone sidebar: ' + error.message + '\n\nCheck the console for more details.');
  }
}

/**
 * Expands the sidebar from the button position
 * @param {HTMLElement} sidebar - The sidebar element
 */
function expandFromButton(sidebar) {
  try {
    debug('Beginning sidebar expansion animation');
    
    // Check if sidebar exists and is in the DOM
    if (!sidebar || !document.body.contains(sidebar)) {
      debug('Sidebar element not found in DOM, cannot expand');
      return;
    }
    
    // Get the close button and ensure it's visible
    const closeButton = sidebar.querySelector('.close-button');
    if (closeButton) {
      closeButton.style.opacity = '1';
      closeButton.style.visibility = 'visible';
    }
    
    // First add a class to begin the transition
    sidebar.classList.add('animating');
    
    // Force a reflow to ensure the initial state is rendered
    void sidebar.offsetHeight;
    
    // Log the current dimensions
    debug('Current sidebar dimensions before expansion:', {
      width: sidebar.offsetWidth,
      height: sidebar.offsetHeight,
      top: sidebar.style.top,
      right: sidebar.style.right
    });
    
    // Then set final dimensions and position
    sidebar.style.width = '380px';
    sidebar.style.height = '100vh';
    sidebar.style.top = '0';
    sidebar.style.right = '0';
    sidebar.style.borderRadius = '0';
    
    debug('Applied expansion styles');
    
    // Add active class after animation completes
    setTimeout(() => {
      debug('Animation complete, adding active class');
      sidebar.classList.remove('animating');
      sidebar.classList.add('active');
      
      // Reset inline styles after animation
      sidebar.style.width = '';
      sidebar.style.height = '';
      sidebar.style.borderRadius = '';
      sidebar.style.backgroundColor = '';
      
      debug('Sidebar expansion complete');
      
      // Ensure close button is visible
      if (closeButton) {
        closeButton.style.opacity = '1';
        closeButton.style.visibility = 'visible';
      }
    }, 500); // Match transition duration
  } catch (error) {
    console.error('DunClone: Error in expandFromButton:', error);
  }
}

/**
 * Collapses the sidebar back to the button
 */
function collapseToButton() {
  try {
    const sidebar = document.getElementById('econ-sidebar');
    if (!sidebar) {
      debug('Sidebar not found for collapse');
      return;
    }
    
    debug('Beginning sidebar collapse animation');
    
    // Add class for animation state
    sidebar.classList.add('animating');
    sidebar.classList.remove('active');
    
    // Set specific styles for collapsing animation
    sidebar.style.top = triggerButtonPosition.top;
    sidebar.style.right = triggerButtonPosition.right;
    sidebar.style.width = '44px';
    sidebar.style.height = '44px';
    sidebar.style.borderRadius = '50%';
    sidebar.style.overflow = 'hidden';
    sidebar.style.backgroundColor = '#0072cf'; // Match button color during animation
    
    debug('Applied collapse styles, waiting for animation to complete');
    
    // Remove the sidebar after animation completes
    setTimeout(() => {
      if (sidebar.parentNode) {
        sidebar.parentNode.removeChild(sidebar);
        sidebarInjected = false;
        debug('Sidebar removed after collapse animation');
      }
    }, 500); // Match transition duration
  } catch (error) {
    console.error('DunClone: Error in collapseToButton:', error);
  }
}

/**
 * Toggles the sidebar visibility
 */
function toggleSidebar() {
  try {
    const sidebar = document.getElementById('econ-sidebar');
    if (!sidebar) {
      debug('Sidebar not found for toggle');
      return;
    }
    
    if (sidebar.classList.contains('active')) {
      debug('Sidebar is active, collapsing');
      collapseToButton();
    } else {
      debug('Sidebar is not active, expanding');
      expandFromButton(sidebar);
    }
  } catch (error) {
    console.error('DunClone: Error in toggleSidebar:', error);
  }
}

/**
 * Sends a request to the background script to get LLM analysis
 * @param {string} objective - The user's research objective
 * @param {string} content - The content to analyze
 */
function requestAnalysis(objective, content) {
  debug('Requesting analysis from background script');
  
  try {
    chrome.runtime.sendMessage({
      type: 'ANALYZE_CONTENT',
      data: {
        objective,
        content
      }
    }, function(response) {
      if (chrome.runtime.lastError) {
        console.error('DunClone: Runtime error:', chrome.runtime.lastError);
        return;
      }
      
      debug('Received analysis response');
      
      // Update the analysis content when we get a response
      const analysisEl = document.getElementById('analysis-content');
      if (analysisEl) {
        if (response && response.analysis) {
          // Format the analysis content with proper HTML structure
          let formattedAnalysis = formatAnalysisContent(response.analysis);
          analysisEl.innerHTML = formattedAnalysis;
          analysisEl.classList.remove('loading');
        } else if (response && response.error) {
          analysisEl.innerHTML = `<div class="error-message"><span class="icon">⚠️</span> Error: ${response.error}</div>`;
          analysisEl.classList.remove('loading');
          analysisEl.classList.add('error');
        }
      } else {
        console.error('DunClone: Analysis element not found');
      }
    });
  } catch (error) {
    console.error('DunClone: Error in requestAnalysis:', error);
  }
}

/**
 * Formats the analysis content with appropriate HTML structure
 * @param {string} analysisText - The raw analysis text from the LLM
 * @returns {string} - Formatted HTML content
 */
function formatAnalysisContent(analysisText) {
  try {
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
  } catch (error) {
    console.error('DunClone: Error formatting analysis content:', error);
    return '<p>Error formatting analysis content.</p>';
  }
}

/**
 * Sends a chat message to the background script
 */
function sendChatMessage() {
  try {
    const chatInputField = document.getElementById('chat-input-field');
    if (!chatInputField) {
      console.error('DunClone: Chat input field not found when trying to send message');
      return;
    }
    
    const message = chatInputField.value.trim();
    
    if (!message) return;
    
    debug('Sending chat message');
    
    // Clear the input field
    chatInputField.value = '';
    
    // Add user message to the chat
    addMessageToChat('user', message);
    
    // Send message to background script
    chrome.runtime.sendMessage({
      type: 'SEND_CHAT_MESSAGE',
      data: {
        message
      }
    }, function(response) {
      if (chrome.runtime.lastError) {
        console.error('DunClone: Runtime error when sending chat message:', chrome.runtime.lastError);
        addMessageToChat('assistant', `Error: Failed to send message - ${chrome.runtime.lastError.message}`);
        return;
      }
      
      if (response && response.response) {
        // Add assistant response to the chat
        addMessageToChat('assistant', response.response);
      } else if (response && response.error) {
        // Add error message to the chat
        addMessageToChat('assistant', `Error: ${response.error}`);
      }
    });
  } catch (error) {
    console.error('DunClone: Error in sendChatMessage:', error);
  }
}

/**
 * Adds a message to the chat UI
 * @param {string} role - 'user' or 'assistant'
 * @param {string} content - The message content
 */
function addMessageToChat(role, content) {
  try {
    const chatMessagesEl = document.getElementById('chat-messages');
    if (!chatMessagesEl) {
      console.error('DunClone: Chat messages element not found');
      return;
    }
    
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
    
    // Scroll to bottom
    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
    
    // Also scroll the content area to show the message
    const contentArea = document.querySelector('.content-area');
    if (contentArea) {
      contentArea.scrollTop = contentArea.scrollHeight;
    }
    
    // Save message to chat history
    chatMessages.push({ role, content });
  } catch (error) {
    console.error('DunClone: Error in addMessageToChat:', error);
  }
}

// Main initialization function
function init() {
  debug('Initializing content script');
  
  try {
    // Check if the page has economics-related content
    isEconomicsRelevant = checkForEconomicsContent();
    
    // If the page is relevant, inject the trigger button
    if (isEconomicsRelevant) {
      debug('Page is economics-relevant, injecting button');
      injectTriggerButton();
    } else {
      debug('Page is not economics-relevant, not injecting button');
    }
    
    // For testing, force inject button after a short delay regardless of content
    if (debugMode) {
      setTimeout(forceShowButton, 2000);
    }
  } catch (error) {
    console.error('DunClone: Error in init:', error);
  }
}

// Add debug info to document load event
document.addEventListener('DOMContentLoaded', function() {
  debug('DOMContentLoaded event fired');
  
  // Run initialization after a short delay to ensure page is fully loaded
  setTimeout(init, 1000);
});

// Run the initialization immediately in case DOMContentLoaded already fired
if (document.readyState === 'interactive' || document.readyState === 'complete') {
  debug('Document already loaded, running init immediately');
  setTimeout(init, 1000);
}

// Emergency fallback - try again after 5 seconds in case init failed
setTimeout(() => {
  if (!triggerButtonInjected) {
    debug('EMERGENCY: No button detected after 5 seconds, forcing injection');
    forceShowButton();
  }
}, 5000);

// Expose debugging functions globally for testing in console
window.DunCloneDebug = {
  forceShowButton,
  injectSidebar,
  toggleSidebar,
  debugMode
};

// Add event listener for the custom test trigger event
document.addEventListener('dunclone_test_trigger', function() {
  debug('Received custom test trigger event');
  forceShowButton();
  
  // Show sidebar after a short delay
  setTimeout(() => {
    injectSidebar();
  }, 500);
});

// Add a direct test method that can be called from browser console
window.testDunClone = async function() {
  debug('Manual test function called from console');
  forceShowButton();
  await new Promise(resolve => setTimeout(resolve, 1000));
  await injectSidebar();
  return "Test complete - check console for details";
};

// Listen for messages from the extension
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  debug('Received message from extension:', message);
  
  if (message.action === 'test_trigger') {
    debug('Received test trigger from extension');
    forceShowButton();
    
    // Show sidebar after a short delay
    setTimeout(() => {
      injectSidebar();
    }, 500);
    
    sendResponse({status: 'success'});
    return true;
  }
  
  return false;
}); console.log("DunClone extension reloaded");
