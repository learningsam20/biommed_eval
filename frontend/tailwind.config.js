export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        clinical: {
          50: '#f0f9f8',
          100: '#d9f0ee',
          200: '#b3e2de',
          300: '#86cdc8',
          400: '#57b1ad',
          500: '#3a9691',
          600: '#2c7a77',
          700: '#276360',
          800: '#23504e',
          900: '#1f4241',
        },
        accent: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        ink: {
          DEFAULT: '#16242c',
          soft: '#3e525c',
          faint: '#6b7f89',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 3px 0 rgb(15 40 44 / 0.08), 0 1px 2px -1px rgb(15 40 44 / 0.06)',
        lift: '0 10px 25px -5px rgb(15 40 44 / 0.15), 0 8px 10px -6px rgb(15 40 44 / 0.08)',
      },
      borderRadius: {
        xl2: '1.25rem',
      },
    },
  },
  plugins: [],
}