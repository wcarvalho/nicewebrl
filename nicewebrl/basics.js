async function isFullscreen() {
  return document.fullscreenElement != null;
} 

async function getImageSeenTime() {
  return window.imageSeenTime;
} 

// Function to send ping to server every 30 seconds
async function pingServer() {
  console.log('Starting ping loop');
  while (true) {
    try {
      const message = `Ping ${Math.floor(Math.random() * 10000)}`;
      // Await the asynchronous emitEvent call
      await emitEvent('ping', { message: message });
      console.log(`Ping sent: ${message}`);
    } catch (err) {
      console.error('Error pinging server:', err);
    }
    // Wait for 30 seconds before sending the next ping
    seconds = 30
    try {
      await new Promise(resolve => setTimeout(resolve, seconds * 1000));
    } catch (err) {
      console.error('Error in ping loop:', err);
    }
  }
}
document.addEventListener('DOMContentLoaded', async function () {

  ////////////////
  // Start pinging the server once the DOM content is fully loaded
  ////////////////
  pingServer();

  ////////////////
  // remove default behavior
  ////////////////
  window.debug = 0;
  window.require_fullscreen = false;
  window.accept_keys = false;
  window.next_states = null;

  ////////////////
  // how to handle key presses?
  ////////////////
  document.addEventListener('keydown', async function (event) {
    // Prevent default behavior for arrow keys
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
      event.preventDefault();
    }

    // Handle key presses
    console.log(event.key);
    if (window.next_states !== null && window.accept_keys && event.key in window.next_states) {
      if (!window.require_fullscreen || await isFullscreen() ) {
        next_state = window.next_states[event.key];
        var imgElement = document.getElementById('stateImage');
        if (imgElement !== null) {
          imgElement.src = next_state;
        }
        window.next_imageSeenTime = new Date();
        console.log('set new image');
      }
      // Record the current time when the keydown event occurs
      var keydownTime = new Date();
      // Await the asynchronous emitEvent call
      await emitEvent('key_pressed', {
        key: event.key,
        keydownTime: keydownTime,
        imageSeenTime: window.imageSeenTime
      });
    }
  });
})