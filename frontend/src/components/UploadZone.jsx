import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileImage, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassPanel from './GlassPanel';

const UploadZone = ({ onFileSelect, selectedFile, onClear }) => {
    const onDrop = useCallback(acceptedFiles => {
        if (acceptedFiles?.length > 0) {
            onFileSelect(acceptedFiles[0]);
        }
    }, [onFileSelect]);

    const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
        onDrop,
        accept: {
            'image/png': ['.png'],
            'image/jpeg': ['.jpg', '.jpeg'],
            'image/tiff': ['.tif', '.tiff']
        },
        maxFiles: 1,
        multiple: false
    });

    return (
        <div className="w-full max-w-xl mx-auto">
            <AnimatePresence mode="wait">
                {!selectedFile ? (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0 }}
                        key="upload-zone"
                    >
                        <div
                            {...getRootProps()}
                            className={`
                relative overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer h-64 flex flex-col items-center justify-center text-center p-8
                ${isDragActive ? 'border-medical-500 bg-medical-50/50' : 'border-clinical-200 hover:border-medical-300 hover:bg-clinical-200/50'}
                ${isDragReject ? 'border-red-400 bg-red-50' : ''}
              `}
                        >
                            <input {...getInputProps()} />

                            <div className="z-10 flex flex-col items-center gap-4">
                                <div className={`p-4 rounded-full ${isDragActive ? 'bg-medical-100 text-medical-600' : 'bg-clinical-50 text-clinical-400'}`}>
                                    <UploadCloud className="w-8 h-8" />
                                </div>

                                <div>
                                    <p className="text-lg font-medium text-clinical-900">
                                        {isDragActive ? 'Drop image here' : 'Click or drop tissue slide'}
                                    </p>
                                    <p className="text-sm text-clinical-500 mt-1">
                                        Supports PNG, JPG, TIF (Max 10MB)
                                    </p>
                                </div>
                            </div>

                            {/* Decorative background grid */}
                            <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
                                style={{ backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)', backgroundSize: '16px 16px' }}>
                            </div>
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        key="file-preview"
                    >
                        <GlassPanel className="flex items-center justify-between p-4 bg-clinical-100/90">
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-12 rounded-lg bg-medical-50 flex items-center justify-center text-medical-600 border border-medical-100">
                                    <FileImage className="w-6 h-6" />
                                </div>
                                <div>
                                    <p className="font-medium text-clinical-900 truncate max-w-[200px]">{selectedFile.name}</p>
                                    <p className="text-xs text-clinical-500">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                                </div>
                            </div>

                            <button
                                onClick={(e) => { e.stopPropagation(); onClear(); }}
                                className="p-2 hover:bg-red-50 text-clinical-400 hover:text-red-500 rounded-lg transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </GlassPanel>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default UploadZone;
