// Switching mailboxes reloads the page so every pane, header and chart
// starts fresh for the new mailbox (and the server remembers the choice).
(function () {
    var select = document.getElementById("mailbox-select");
    if (!select) {
        return;
    }
    select.addEventListener("change", function () {
        window.location.assign("/?mailbox=" + encodeURIComponent(select.value));
    });
})();
