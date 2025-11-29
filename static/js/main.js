// main.js

// Auto-hide flash messages after a few seconds
window.addEventListener("DOMContentLoaded", () => {
    const flashes = document.querySelectorAll(".flash-message");
    if (!flashes.length) return;

    setTimeout(() => {
        flashes.forEach(msg => {
            msg.style.transition = "opacity 0.4s ease, transform 0.4s ease";
            msg.style.opacity = "0";
            msg.style.transform = "translateY(-5px)";
            setTimeout(() => msg.remove(), 500);
        });
    }, 3500);
});
