        let htmlInput = document.getElementById('htmlInput');
        let iframe = document.getElementById('contentFrame');
        const heightDisplay = document.getElementById('iframeHeight');

        // Load content into iframe
        function loadContent(html_content) {
            const htmlContent = htmlInput.value || '<p style="padding: 20px;">Enter some HTML content and click "Load Content"</p>';

            // Wrap content with resize script
            const wrappedContent = `
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body { margin: 0; padding: 16px; font-family: system-ui, -apple-system, sans-serif; }
                    </style>
                </head>
                <body>
                    ${htmlContent}
                    <script>
                        // Send height updates to parent
                        function sendHeight() {
                            try {
                                const height = Math.max(
                                    document.body.scrollHeight,
                                    document.documentElement.scrollHeight,
                                    document.body.offsetHeight,
                                    document.documentElement.offsetHeight
                                );
                                console.log(document.body.scrollHeight);
                                console.log(document.documentElement.scrollHeight);
                                console.log(document.body.offsetHeight);
                                console.log(document.documentElement.offsetHeight);
                                console.log('just some print to enter the debugger');

                                window.parent.postMessage({
                                    type: 'resize',
                                    height: height
                                }, '*');

                                console.log('Sent height:', height);
                            } catch(e) {
                                console.error('Failed to send height:', e);
                            }
                        }

                        // Wait for DOM to be ready
                        if (document.readyState === 'loading') {
                            document.addEventListener('DOMContentLoaded', sendHeight);
                        } else {
                            sendHeight();
                        }

                        // Send after a short delay to ensure rendering
                        setTimeout(sendHeight, 50);
                        setTimeout(sendHeight, 200);
                        setTimeout(sendHeight, 500);

                        // Watch for changes
                        const observer = new MutationObserver(function() {
                            setTimeout(sendHeight, 10);
                        });

                        observer.observe(document.body, {
                            attributes: true,
                            childList: true,
                            subtree: true
                        });

                        // Watch for resize
                        window.addEventListener('resize', sendHeight);

                        // Watch for images loading
                        setTimeout(function() {
                            document.querySelectorAll('img').forEach(img => {
                                if (!img.complete) {
                                    img.addEventListener('load', sendHeight);
                                }
                            });
                        }, 0);
                    <\/script>
                </body>
                </html>
            `;

            iframe.srcdoc = wrappedContent;
        }

         // Listen for resize messages from iframe
        window.addEventListener('message', function(event) {
            // Add safety check for event.data
            if (event.data && event.data.type === 'resize') {
                    const height = event.data.height;
                    iframe.style.height = height + 'px';
                    heightDisplay.textContent = `Height: ${height}px`;
            }
        }, false);

        // Also handle iframe load event
        iframe.addEventListener('load', function() {
            // Give iframe time to calculate height
            setTimeout(function() {
                // Trigger a manual check if postMessage fails
                try {
                    const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                    if (iframeDoc) {
                        const height = iframeDoc.documentElement.scrollHeight;
                        iframe.style.height = height + 'px';
                        heightDisplay.textContent = `Height: ${height}px (direct access)`;
                    }
                } catch(e) {
                    console.log('Direct access blocked (cross-origin), relying on postMessage');
                }
            }, 100);
        });