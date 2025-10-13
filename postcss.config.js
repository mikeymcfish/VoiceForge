export default {
  // Ensure PostCSS has a `from` value to avoid warnings from plugins
  // that call `postcss.parse` without providing it.
  // Using `undefined` tells PostCSS there is no input file on disk.
  options: { from: undefined },
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
