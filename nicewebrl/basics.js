function isFullscreen() {
  return document.fullscreenElement !== null;
} 

document.addEventListener('DOMContentLoaded', async function () {

  ////////////////
  // remove default behavior
  ////////////////
  window.debug = 0;
  document.addEventListener('keydown', function (event) {
    switch (event.key) {
      case "ArrowUp":
      case "ArrowDown":
      case "ArrowLeft":
      case "ArrowRight":
        event.preventDefault();
        break;
      default:
        break;
    }
  });

  ////////////////
  // how to handle key presses?
  ////////////////
  document.addEventListener('keydown', function (event) {

    console.log(event.key)
    if (event.key in window.next_states) {
      if (isFullscreen() || window.debug > 0){
        next_state = window.next_states[event.key];
        var imgElement = document.getElementById('stateImage')
        imgElement.src = next_state;
        window.next_imageSeenTime = new Date();
        console.log('set new image')
      }
      // Record the current time when the keydown event occurs
      var keydownTime = new Date();
      // Emit the keydown event with the latest times
      emitEvent('key_pressed', {
        key: event.key,
        keydownTime: keydownTime,
        imageSeenTime: window.imageSeenTime
      });
    }
  });
})