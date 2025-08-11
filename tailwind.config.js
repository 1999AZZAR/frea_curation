/**** Tailwind Config ****/
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/**/*.html',
    './static/js/**/*.js',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['ui-sans-serif','system-ui','-apple-system','Segoe UI','Roboto','Inter','Ubuntu','Cantarell','Noto Sans','sans-serif']
      },
      colors: {
        slate: {
          950: '#020617'
        }
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(99,102,241,.15), 0 10px 30px -12px rgba(0,0,0,.6)'
      }
    }
  },
  plugins: []
}
