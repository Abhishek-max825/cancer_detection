import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const ClinicalButton = ({
    children,
    variant = 'primary',
    className,
    isLoading = false,
    icon: Icon,
    ...props
}) => {
    const baseStyles = "btn-clinical relative overflow-hidden group";

    const variants = {
        primary: "bg-medical-500 text-white hover:bg-medical-600 shadow-md shadow-medical-500/20",
        secondary: "bg-white text-clinical-100 border border-clinical-100 hover:bg-clinical-50 hover:text-clinical-900 hover:border-clinical-200",
        outline: "bg-transparent border border-medical-500 text-medical-500 hover:bg-medical-50",
        ghost: "bg-transparent text-clinical-500 hover:text-medical-600 hover:bg-medical-50/50",
        danger: "bg-red-50 text-red-600 hover:bg-red-100 border border-red-100"
    };

    return (
        <motion.button
            whileTap={{ scale: 0.98 }}
            className={twMerge(baseStyles, variants[variant], className)}
            {...props}
        >
            {isLoading ? (
                <span className="flex items-center gap-2">
                    <svg className="animate-spin h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                </span>
            ) : (
                <span className="flex items-center gap-2">
                    {Icon && <Icon className="w-4 h-4 transition-transform group-hover:scale-110" />}
                    {children}
                </span>
            )}
        </motion.button>
    );
};

export default ClinicalButton;
