// Loaded once per page. Email bodies render in sandboxed iframes (opaque origin),
// which report their content height via postMessage; see email_render.py.
(function () {
    window.addEventListener("message", function (event) {
        var data = event.data;
        if (!data || data.type !== "mbox-frame-height" || typeof data.height !== "number") {
            return;
        }
        var frames = document.querySelectorAll("iframe.email-frame");
        for (var i = 0; i < frames.length; i++) {
            if (frames[i].contentWindow === event.source) {
                frames[i].style.height = Math.max(40, Math.min(data.height, 100000)) + "px";
                return;
            }
        }
    });

    document.addEventListener("click", function (event) {
        var item = event.target.closest(".email-list-item");
        if (!item) {
            return;
        }
        document.querySelectorAll(".email-list-item.is-selected").forEach(function (el) {
            el.classList.remove("is-selected");
        });
        item.classList.add("is-selected");
    });
})();
