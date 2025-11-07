/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                netflix: {
                    red: '#E50914',
                    black: '#141414',
                    darkGray: '#181818',
                    gray: '#2F2F2F',
                    lightGray: '#808080',
                },
                light: {
                    bg: '#FFFFFF',
                    card: '#F9FAFB',
                    border: '#E5E7EB',
                    text: '#111827',
                    textSecondary: '#6B7280',
                },
            },
            fontFamily: {
                netflix: ['Netflix Sans', 'Helvetica Neue', 'Segoe UI', 'Roboto', 'sans-serif'],
            },
        },
    },
    plugins: [],
}