document.addEventListener('DOMContentLoaded', function() {
  // Get the DOM elements
  const researchObjective = document.getElementById('researchObjective');
  const saveButton = document.getElementById('saveButton');
  const statusEl = document.getElementById('status');
  
  // Add debug UI elements
  addDebugControls();
  
  // Load any existing research objective from storage
  chrome.storage.local.get(['researchObjective'], function(result) {
    if (result.researchObjective) {
      researchObjective.value = result.researchObjective;
    }
  });
  
  // Add click event listener to the save button
  saveButton.addEventListener('click', function() {
    const objective = researchObjective.value.trim();
    
    if (!objective) {
      statusEl.textContent = 'Please enter a research objective';
      statusEl.style.color = '#dc3545';
      return;
    }
    
    // Save the research objective to chrome.storage
    chrome.storage.local.set({
      researchObjective: objective
    }, function() {
      // Update the status to show it was saved
      statusEl.textContent = 'Research objective saved!';
      statusEl.style.color = '#28a745';
      
      // Clear the status message after 2 seconds
      setTimeout(function() {
        statusEl.textContent = '';
      }, 2000);
    });
  });
});

/**
 * Adds debug controls to the popup for testing
 */
function addDebugControls() {
  // Create debug controls container
  const debugContainer = document.createElement('div');
  debugContainer.className = 'debug-container';
  debugContainer.innerHTML = `
    <h3>Debug Controls</h3>
    <div class="debug-actions">
      <button id="testButton" class="button debug-button">Test in Current Tab</button>
      <button id="openTestPage" class="button debug-button">Open Test Page</button>
      <button id="openSimpleTest" class="button debug-button">Simple Test</button>
    </div>
    <div id="debugStatus" class="status"></div>
  `;
  
  // Add styles
  const style = document.createElement('style');
  style.textContent = `
    .debug-container {
      margin-top: 20px;
      padding-top: 20px;
      border-top: 1px solid #e1e4e8;
    }
    .debug-container h3 {
      color: #6c757d;
      margin-top: 0;
    }
    .debug-actions {
      display: flex;
      gap: 10px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    .debug-button {
      flex: 1;
      background-color: #6c757d;
      min-width: 120px;
    }
    .debug-button:hover {
      background-color: #5a6268;
    }
  `;
  
  // Add to document
  document.querySelector('.content').appendChild(debugContainer);
  document.head.appendChild(style);
  
  // Add event listeners
  document.getElementById('testButton').addEventListener('click', function() {
    // Get the current active tab
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const currentTab = tabs[0];
      if (currentTab) {
        // Execute script in the current tab to trigger DunClone
        chrome.scripting.executeScript({
          target: {tabId: currentTab.id},
          function: triggerDunClone
        });
        
        // Update debug status
        const debugStatus = document.getElementById('debugStatus');
        debugStatus.textContent = 'Test triggered in current tab';
        debugStatus.style.color = '#28a745';
        
        // Close the popup
        setTimeout(() => window.close(), 1000);
      }
    });
  });
  
  document.getElementById('openTestPage').addEventListener('click', function() {
    // Get the extension's URL for the test page
    const testPageUrl = chrome.runtime.getURL('test.html');
    
    // Open the test page in a new tab
    chrome.tabs.create({url: testPageUrl});
    
    // Close the popup
    window.close();
  });
  
  // Add event listener for simple test page
  document.getElementById('openSimpleTest').addEventListener('click', function() {
    // Get the extension's URL for the simple test page
    const simpleTestUrl = chrome.runtime.getURL('simple-test.html');
    
    // Open the simple test page in a new tab
    chrome.tabs.create({url: simpleTestUrl});
    
    // Close the popup
    window.close();
  });
}

/**
 * This function will be injected into the current tab
 * to test the DunClone functionality
 */
function triggerDunClone() {
  console.log('DunClone test triggered from popup');
  
  // Check if the debug object exists
  if (window.DunCloneDebug) {
    // Force show the button
    window.DunCloneDebug.forceShowButton();
    
    // Show sidebar after a short delay
    setTimeout(() => {
      if (window.DunCloneDebug.injectSidebar) {
        window.DunCloneDebug.injectSidebar();
      }
    }, 1000);
    
    return 'DunClone test triggered successfully';
  } else {
    console.error('DunClone debug object not found, extension might not be initialized');
    
    // Try to dispatch a custom event to trigger the extension
    const event = new CustomEvent('dunclone_test_trigger');
    document.dispatchEvent(event);
    
    return 'DunClone debug object not found, attempted event dispatch';
  }
} 