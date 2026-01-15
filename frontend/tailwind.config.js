/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                fragile: {
                    50: '#F8F9FA',  // Ultra-light gray/white
                    100: '#F1F3F5', // Very light gray
                    200: '#E9ECEF', // Light gray
                    300: '#DEE2E6', // Soft gray border
                    400: '#CED4DA', // Medium soft gray
                    500: '#ADB5BD', // Muted text
                    600: '#868E96', // Secondary text
                    700: '#495057', // Primary text (soft)
                    800: '#343A40', // Dark text (soft)
                    900: '#212529', // Darkest text
                },
                delicate: {
                    blue: '#D0EBFF',
                    purple: '#E5DBFF',
                    red: '#FFD5D5',
                    green: '#D3F9D8',
                }
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            fontWeight: {
                thin: '100',
                extralight: '200',
                light: '300',
            },
            animation: {
                'fade-in': 'fadeIn 1.2s ease-out forwards',
                'float': 'float 6s ease-in-out infinite',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0', transform: 'translateY(10px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                float: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-10px)' },
                }
            }
        },
    },
    plugins: [],
}
