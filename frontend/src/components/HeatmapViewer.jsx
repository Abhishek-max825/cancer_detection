import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, EyeOff, Info, SplitSquareHorizontal, Layers } from 'lucide-react';
import { clsx } from 'clsx';

const HeatmapViewer = ({ originalImage, heatmapImage, prediction, confidence, entropy, patternType, ensembleVotes }) => {
    const [viewMode, setViewMode] = useState('side-by-side'); // 'side-by-side' or 'overlay'
    const [showOverlay, setShowOverlay] = useState(true);

    const isPositive = prediction === 'Cancer';
    const confidencePercent = (confidence * 100).toFixed(1);

    // Borderline detection: 45% to 55%
    const isBorderline = confidence > 0.45 && confidence < 0.55;

    // Helper for dynamic colors/text
    const getStatusColor = () => {
        if (isBorderline) return {
            text: "text-amber-500",
            bg: "bg-amber-50",
            border: "border-amber-100",
            gradient: "from-amber-500 to-orange-400",
            glow: "from-amber-500 to-orange-500",
            pulse: "bg-amber-500"
        };
        return isPositive ? {
            text: "text-red-600",
            bg: "bg-red-50",
            border: "border-red-100",
            gradient: "from-red-500 to-rose-400",
            glow: "from-red-500 to-rose-500",
            pulse: "bg-red-500"
        } : {
            text: "text-emerald-600",
            bg: "bg-emerald-50",
            border: "border-emerald-100",
            gradient: "from-emerald-500 to-teal-400",
            glow: "from-emerald-500 to-teal-500",
            pulse: "bg-emerald-500"
        };
    };

    const status = getStatusColor();
    const label = isBorderline ? "Ambiguous" : (isPositive ? "Malignant" : "Benign");

    // Zoom state
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

    const handleZoomIn = () => setZoom(z => Math.min(z + 0.5, 4));
    const handleZoomOut = () => setZoom(z => Math.max(z - 0.5, 1));
    const handleResetZoom = () => {
        setZoom(1);
        setPan({ x: 0, y: 0 });
    };

    const handleMouseDown = (e) => {
        if (zoom > 1) {
            setIsDragging(true);
            setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
        }
    };

    const handleMouseMove = (e) => {
        if (isDragging) {
            setPan({
                x: e.clientX - dragStart.x,
                y: e.clientY - dragStart.y
            });
        }
    };

    const handleMouseUp = () => setIsDragging(false);

    return (
        <div className="w-full space-y-6">
            {/* View Controls */}
            <div className="flex justify-between items-center bg-clinical-50 p-2 rounded-xl border border-clinical-100">
                <div className="flex gap-2">
                    <button
                        onClick={handleZoomIn}
                        className="p-2 rounded-lg bg-clinical-200 border border-clinical-300 text-clinical-500 hover:text-medical-500 hover:border-medical-500 transition-all shadow-sm"
                        title="Zoom In"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="11" y1="8" x2="11" y2="14" /><line x1="8" y1="11" x2="14" y2="11" /></svg>
                    </button>
                    <button
                        onClick={handleZoomOut}
                        disabled={zoom <= 1}
                        className="p-2 rounded-lg bg-clinical-200 border border-clinical-300 text-clinical-500 hover:text-medical-500 hover:border-medical-500 transition-all shadow-sm disabled:opacity-50"
                        title="Zoom Out"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="8" y1="11" x2="14" y2="11" /></svg>
                    </button>
                    <button
                        onClick={handleResetZoom}
                        disabled={zoom === 1}
                        className="p-2 rounded-lg bg-clinical-200 border border-clinical-300 text-clinical-500 hover:text-medical-500 hover:border-medical-500 transition-all shadow-sm disabled:opacity-50"
                        title="Reset View"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /></svg>
                    </button>
                    <span className="px-3 py-2 text-xs font-mono text-clinical-400 flex items-center">
                        ZOOM: {Math.round(zoom * 100)}%
                    </span>
                </div>

                <div className="flex gap-2">
                    <button
                        onClick={() => setViewMode('side-by-side')}
                        className={clsx(
                            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all",
                            viewMode === 'side-by-side'
                                ? "bg-clinical-200 text-clinical-900 shadow-sm ring-1 ring-clinical-300"
                                : "text-clinical-500 hover:bg-clinical-200/50"
                        )}
                    >
                        <SplitSquareHorizontal className="w-4 h-4" />
                        Side-by-Side
                    </button>
                    <button
                        onClick={() => setViewMode('overlay')}
                        className={clsx(
                            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all",
                            viewMode === 'overlay'
                                ? "bg-clinical-200 text-clinical-900 shadow-sm ring-1 ring-clinical-300"
                                : "text-clinical-500 hover:bg-clinical-200/50"
                        )}
                    >
                        <Layers className="w-4 h-4" />
                        Overlay
                    </button>
                </div>
            </div>

            {/* Image Display Area */}
            <div className="relative w-full overflow-hidden rounded-2xl bg-black/5 border border-clinical-100 shadow-inner p-1">
                <AnimatePresence mode="wait">
                    {viewMode === 'side-by-side' ? (
                        <motion.div
                            key="side-by-side"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="grid grid-cols-1 md:grid-cols-2 gap-4"
                        >
                            {/* Original */}
                            <div className="space-y-2">
                                <span className="text-xs font-semibold text-clinical-400 uppercase tracking-wider pl-1 flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-clinical-300"></div>
                                    Original Tissue
                                </span>
                                <div
                                    className="aspect-square rounded-xl overflow-hidden shadow-sm border border-white/20 bg-white relative cursor-move"
                                    onMouseDown={handleMouseDown}
                                    onMouseMove={handleMouseMove}
                                    onMouseUp={handleMouseUp}
                                    onMouseLeave={handleMouseUp}
                                >
                                    <div
                                        style={{
                                            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                                            transformOrigin: 'center',
                                            transition: isDragging ? 'none' : 'transform 0.2s ease-out'
                                        }}
                                        className="w-full h-full"
                                    >
                                        <img src={originalImage} alt="Original" className="w-full h-full object-cover" />
                                    </div>
                                </div>
                            </div>
                            {/* Heatmap */}
                            <div className="space-y-2">
                                <span className="text-xs font-semibold text-clinical-400 uppercase tracking-wider pl-1 flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-medical-500"></div>
                                    AI Attention Map
                                </span>
                                <div
                                    className="aspect-square rounded-xl overflow-hidden shadow-sm border border-white/20 bg-white relative cursor-move"
                                    onMouseDown={handleMouseDown}
                                    onMouseMove={handleMouseMove}
                                    onMouseUp={handleMouseUp}
                                    onMouseLeave={handleMouseUp}
                                >
                                    <div
                                        style={{
                                            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                                            transformOrigin: 'center',
                                            transition: isDragging ? 'none' : 'transform 0.2s ease-out'
                                        }}
                                        className="w-full h-full"
                                    >
                                        <img src={heatmapImage} alt="Heatmap" className="w-full h-full object-cover" />
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="overlay"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="aspect-square w-full max-w-md mx-auto relative rounded-xl overflow-hidden shadow-sm border border-white/20 bg-white cursor-move"
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                        >
                            <div
                                style={{
                                    transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                                    transformOrigin: 'center',
                                    transition: isDragging ? 'none' : 'transform 0.2s ease-out'
                                }}
                                className="w-full h-full relative"
                            >
                                <img
                                    src={showOverlay ? heatmapImage : originalImage}
                                    alt="Analysis View"
                                    className="absolute inset-0 w-full h-full object-cover"
                                />
                            </div>

                            <button
                                onClick={(e) => { e.stopPropagation(); setShowOverlay(!showOverlay); }}
                                className="absolute bottom-4 right-4 z-20 bg-white/90 backdrop-blur-md p-2 rounded-full shadow-sm text-clinical-600 hover:text-medical-600 transition-colors border border-clinical-100"
                            >
                                {showOverlay ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Analysis Results Card */}
            <div className="bg-clinical-100/50 backdrop-blur-xl border border-white/10 shadow-xl rounded-2xl p-6 relative overflow-hidden">
                {/* Decorative background glow */}
                <div className={clsx(
                    "absolute top-0 right-0 w-64 h-64 bg-gradient-to-br opacity-10 blur-3xl rounded-full -mr-16 -mt-16 pointer-events-none",
                    status.glow
                )} />

                <div className="relative z-10 grid grid-cols-1 lg:grid-cols-4 gap-6 items-center">
                    <div className="flex items-center justify-between col-span-1">
                        <div>
                            <h3 className="text-sm font-medium text-clinical-500 uppercase tracking-wider">Prediction</h3>
                            <div className="flex flex-col mt-1">
                                <span className={clsx(
                                    "text-3xl font-display font-bold leading-tight",
                                    status.text
                                )}>
                                    {label}
                                </span>
                                <span className="text-clinical-400 text-sm font-medium uppercase tracking-wider">
                                    {prediction}
                                </span>
                            </div>
                        </div>
                        <div className={clsx(
                            "w-8 h-8 shrink-0 rounded-full flex items-center justify-center border-2",
                            status.border, status.bg, status.text
                        )}>
                            <div className={clsx("w-2 h-2 rounded-full animate-pulse", status.pulse)} />
                        </div>
                    </div>

                    <div className="space-y-4 col-span-1 lg:col-span-1 border-l border-clinical-200 pl-6">
                        {/* Pattern Type */}
                        <div>
                            <span className="text-xs font-semibold text-clinical-400 uppercase tracking-wider block mb-1">
                                Pattern Analysis
                            </span>
                            <span className="text-sm font-medium text-clinical-700 bg-white/50 px-2 py-1 rounded-md border border-clinical-100">
                                {patternType || "N/A"}
                            </span>
                        </div>

                        {/* Uncertainty / Entropy */}
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-clinical-400 font-semibold uppercase tracking-wider">Model Uncertainty</span>
                                <span className="text-clinical-600 font-mono">
                                    {entropy ? (entropy * 100).toFixed(1) : 0}%
                                </span>
                            </div>
                            <div className="h-1.5 w-full bg-clinical-100 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${entropy ? entropy * 100 : 0}%` }}
                                    className="h-full bg-slate-400 rounded-full"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="space-y-2 col-span-1 lg:col-span-1 border-l border-clinical-200 pl-6">
                        <div className="flex justify-between text-sm">
                            <span className="text-clinical-600">Confidence Score</span>
                            <span className="font-semibold text-clinical-900">{confidencePercent}%</span>
                        </div>
                        <div className="h-3 w-full bg-clinical-100 rounded-full overflow-hidden p-[2px]">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${confidencePercent}%` }}
                                transition={{ duration: 1.2, ease: "easeOut" }}
                                className={clsx(
                                    "h-full rounded-full shadow-sm bg-gradient-to-r",
                                    status.gradient
                                )}
                            />
                        </div>
                    </div>

                    <div className="bg-medical-500/10 border border-medical-500/20 p-4 rounded-xl flex gap-3 text-sm text-medical-100 col-span-1">
                        <Info className="w-5 h-5 shrink-0 text-medical-500" />
                        <p className="leading-relaxed text-xs">
                            {isBorderline ? (
                                <>
                                    Uncertainty detected.
                                    <span className="block mt-1 font-medium text-amber-500 opacity-100">Expert review required.</span>
                                </>
                            ) : (
                                <>
                                    AI analysis complete.
                                    <span className="block mt-1 font-medium opacity-80">Correlation with {patternType} pattern.</span>
                                </>
                            )}
                        </p>
                    </div>

                    {/* Ensemble Jury Vote */}
                    {ensembleVotes && ensembleVotes.length > 0 && (
                        <div className="col-span-1 lg:col-span-4 mt-6 pt-6 border-t border-clinical-200">
                            <h4 className="text-xs font-semibold text-clinical-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-clinical-400"></div>
                                Ensemble Jury Breakdown
                            </h4>
                            <div className="grid grid-cols-3 gap-4">
                                {['DenseNet121', 'ResNet50', 'EfficientNet'].map((modelName, idx) => {
                                    const vote = ensembleVotes[idx] || 0;
                                    const percentage = (vote * 100).toFixed(1);
                                    const isHigh = vote > 0.5;

                                    return (
                                        <div key={modelName} className="bg-clinical-50/80 rounded-lg p-3 border border-clinical-200 shadow-sm flex flex-col gap-2 backdrop-blur-sm">
                                            <div className="flex justify-between items-center text-xs">
                                                <span className="text-clinical-700 font-bold">{modelName}</span>
                                                <span className={clsx("font-mono font-bold text-sm", isHigh ? "text-rose-600" : "text-emerald-600")}>
                                                    {percentage}%
                                                </span>
                                            </div>
                                            <div className="h-2 w-full bg-white border border-clinical-100 rounded-full overflow-hidden">
                                                <motion.div
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${percentage}%` }}
                                                    transition={{ delay: 0.2 + (idx * 0.1) }}
                                                    className={clsx(
                                                        "h-full rounded-full shadow-sm",
                                                        isHigh ? "bg-gradient-to-r from-rose-500 to-red-600" : "bg-gradient-to-r from-emerald-400 to-teal-500"
                                                    )}
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default HeatmapViewer;
