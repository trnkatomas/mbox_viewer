/**
 * The Docker build (see Dockerfile) turns this into static/vendor/tailwind.css,
 * keeping only the classes found in these files. Class names must appear in
 * full in the source: one assembled at runtime (e.g. "text-" + color) is missed.
 */
module.exports = {
  content: ["./templates/**/*.{html,jinja}", "./static/*.js", "./*.py"],
  theme: { extend: {} },
  plugins: [],
};
