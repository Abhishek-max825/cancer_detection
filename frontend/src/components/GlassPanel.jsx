import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const GlassPanel = ({ children, className, hoverEffect = false, ...props }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className={twMerge(
                "glass-panel p-6",
                hoverEffect && "glass-panel-hover",
                className
            )}
            {...props}
        >
            {children}
        </motion.div>
    );
};

export default GlassPanel;
