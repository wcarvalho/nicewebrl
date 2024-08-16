function isFullscreen() {
  return document.fullscreenElement !== null;
} 

document.addEventListener('DOMContentLoaded', async function () {

  ////////////////
  // remove default behavior
  ////////////////
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
      if (isFullscreen()){
        next_state = window.next_states[event.key];
        var imgElement = document.getElementById('stateImage')
        imgElement.src = next_state;
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