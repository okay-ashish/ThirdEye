/**
 * ThirdEye System — Phase 3
 * Forensic Composite Sketch Constructor Client Controller
 * Manages Category-First workflow, Fabric.js canvas, layer ordering,
 * alignment guides, snap-to-center, skin-tone selection, and reconstruction dispatch.
 */

document.addEventListener('DOMContentLoaded', function () {
    const canvasElement = document.getElementById('sketchCanvas');
    if (!canvasElement) return;

    // 1. Initialize Fabric.js Canvas
    const canvas = new fabric.Canvas('sketchCanvas', {
        backgroundColor: '#ffffff',
        preserveObjectStacking: true,
        selectionColor: 'rgba(0, 206, 201, 0.15)',
        selectionBorderColor: '#00CEC9',
        selectionLineWidth: 1.5
    });

    // Custom object transformation styling
    fabric.Object.prototype.transparentCorners = false;
    fabric.Object.prototype.cornerColor = '#0984E3';
    fabric.Object.prototype.cornerStrokeColor = '#00CEC9';
    fabric.Object.prototype.cornerSize = 10;
    fabric.Object.prototype.borderColor = '#00CEC9';
    fabric.Object.prototype.cornerStyle = 'circle';

    let centerGuideLine = null;
    let isGuideVisible = false;
    let isSnapEnabled = true;

    // Category display metadata
    const categoryMeta = {
        'head': { title: 'Head Shapes', icon: '👤', defaultTop: 70, defaultScale: 0.72, layer: 'bottom' },
        'hair': { title: 'Hairstyles', icon: '💇', defaultTop: 20, defaultScale: 0.65, layer: 'top' },
        'eyebrows': { title: 'Eyebrows', icon: '〰️', defaultTop: 180, defaultScale: 0.58, layer: 'middle' },
        'eyes': { title: 'Eyes', icon: '👁️', defaultTop: 215, defaultScale: 0.58, layer: 'middle' },
        'nose': { title: 'Nose Structures', icon: '👃', defaultTop: 260, defaultScale: 0.55, layer: 'middle' },
        'lips': { title: 'Lips & Mouth', icon: '👄', defaultTop: 345, defaultScale: 0.55, layer: 'middle' },
        'mustach': { title: 'Facial Hair / Beard', icon: '🧔', defaultTop: 320, defaultScale: 0.55, layer: 'middle' },
        'more': { title: 'Glasses & Accessories', icon: '👓', defaultTop: 210, defaultScale: 0.58, layer: 'top' }
    };

    // --- 2. Category Navigation Filtering ---
    const categoryButtons = document.querySelectorAll('.cat-btn');
    const categoryGroups = document.querySelectorAll('.category-items-group');
    const titleDisplay = document.getElementById('categoryTitleDisplay');
    const countDisplay = document.getElementById('categoryCountDisplay');
    const badgeDisplay = document.getElementById('activeCategoryBadge');

    categoryButtons.forEach(btn => {
        btn.addEventListener('click', function () {
            categoryButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            const cat = this.dataset.category;
            categoryGroups.forEach(grp => {
                grp.style.display = (grp.id === `group-${cat}`) ? 'block' : 'none';
            });

            const meta = categoryMeta[cat] || { title: cat.toUpperCase(), icon: '📁' };
            if (titleDisplay) titleDisplay.textContent = meta.title;
            if (badgeDisplay) badgeDisplay.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);

            const activeGroup = document.getElementById(`group-${cat}`);
            const count = activeGroup ? activeGroup.querySelectorAll('.part-card').length : 0;
            if (countDisplay) countDisplay.textContent = `${count} Items`;
        });
    });

    // --- 3. Adding Parts to Canvas with Smart Placement ---
    document.querySelectorAll('.part-card').forEach(card => {
        card.addEventListener('click', function () {
            const partSrc = this.dataset.part;
            const category = this.dataset.category || 'misc';
            const meta = categoryMeta[category] || { defaultTop: 100, defaultScale: 0.5 };

            fabric.Image.fromURL(partSrc, function (oImg) {
                if (!oImg) return;

                oImg.scale(meta.defaultScale);
                oImg.set({
                    category: category,
                    originX: 'center',
                    originY: 'top',
                    left: canvas.getWidth() / 2,
                    top: meta.defaultTop
                });

                canvas.add(oImg);

                // Smart initial layer placement
                if (meta.layer === 'bottom') {
                    oImg.sendToBack();
                } else if (meta.layer === 'top') {
                    oImg.bringToFront();
                }

                canvas.setActiveObject(oImg);
                canvas.renderAll();
                updateLayerControls();
            }, { crossOrigin: 'anonymous' });
        });
    });

    // --- 4. Layer Ordering & Object Selection Handlers ---
    const layerControlsBar = document.getElementById('layerControlsBar');

    function updateLayerControls() {
        const active = canvas.getActiveObject();
        if (layerControlsBar) {
            if (active && active !== centerGuideLine) {
                layerControlsBar.classList.remove('disabled');
            } else {
                layerControlsBar.classList.add('disabled');
            }
        }
    }

    canvas.on('selection:created', updateLayerControls);
    canvas.on('selection:updated', updateLayerControls);
    canvas.on('selection:cleared', updateLayerControls);

    document.getElementById('bringForwardBtn')?.addEventListener('click', function () {
        const active = canvas.getActiveObject();
        if (active) {
            canvas.bringForward(active);
            canvas.renderAll();
        }
    });

    document.getElementById('sendBackwardBtn')?.addEventListener('click', function () {
        const active = canvas.getActiveObject();
        if (active) {
            canvas.sendBackwards(active);
            canvas.renderAll();
        }
    });

    document.getElementById('bringToFrontBtn')?.addEventListener('click', function () {
        const active = canvas.getActiveObject();
        if (active) {
            canvas.bringToFront(active);
            canvas.renderAll();
        }
    });

    document.getElementById('sendToBackBtn')?.addEventListener('click', function () {
        const active = canvas.getActiveObject();
        if (active) {
            canvas.sendToBack(active);
            canvas.renderAll();
        }
    });

    // --- 5. Delete Object ---
    function deleteActiveObject() {
        const active = canvas.getActiveObject();
        if (active && active !== centerGuideLine) {
            canvas.remove(active);
            canvas.discardActiveObject();
            canvas.renderAll();
            updateLayerControls();
        }
    }

    document.getElementById('deleteBtn')?.addEventListener('click', deleteActiveObject);

    window.addEventListener('keydown', function (e) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            // Only trigger if not typing inside an input
            if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                deleteActiveObject();
            }
        }
    });

    // --- 6. Alignment Guides & Snap to Center ---
    const toggleGuideBtn = document.getElementById('toggleGuideBtn');
    const snapToCenterBtn = document.getElementById('snapToCenterBtn');

    function createOrUpdateGuide() {
        const midX = canvas.getWidth() / 2;
        if (!centerGuideLine) {
            centerGuideLine = new fabric.Line([midX, 0, midX, canvas.getHeight()], {
                stroke: 'rgba(0, 206, 201, 0.45)',
                strokeWidth: 1.5,
                strokeDashArray: [6, 4],
                selectable: false,
                evented: false,
                excludeFromExport: true
            });
            canvas.add(centerGuideLine);
        }
        centerGuideLine.set('visible', isGuideVisible);
        centerGuideLine.bringToFront();
        canvas.renderAll();
    }

    toggleGuideBtn?.addEventListener('click', function () {
        isGuideVisible = !isGuideVisible;
        this.classList.toggle('active', isGuideVisible);
        createOrUpdateGuide();
    });

    snapToCenterBtn?.addEventListener('click', function () {
        isSnapEnabled = !isSnapEnabled;
        this.classList.toggle('active', isSnapEnabled);
    });

    // Snap to vertical center axis during movement
    canvas.on('object:moving', function (e) {
        if (!isSnapEnabled) return;
        const obj = e.target;
        if (!obj || obj === centerGuideLine) return;

        const canvasCenter = canvas.getWidth() / 2;
        const objCenter = obj.getCenterPoint().x;
        const threshold = 12;

        if (Math.abs(objCenter - canvasCenter) < threshold) {
            obj.setPositionByOrigin(new fabric.Point(canvasCenter, obj.getCenterPoint().y), 'center', 'center');
            obj.setCoords();
        }
    });

    // --- 7. Load Base Face Preset ---
    document.getElementById('loadDefaultFaceBtn')?.addEventListener('click', function () {
        if (canvas.getObjects().filter(o => o !== centerGuideLine).length > 0) {
            if (!confirm('Load basic base face? This will clear current additions.')) return;
        }

        canvas.clear();
        canvas.backgroundColor = '#ffffff';
        centerGuideLine = null;
        if (isGuideVisible) createOrUpdateGuide();

        // Sequential loading of standard forensic base
        const parts = [
            { path: '/static/sketch_parts/head/01.png', top: 70, scale: 0.72, layer: 'bottom' },
            { path: '/static/sketch_parts/hair/01.png', top: 20, scale: 0.65, layer: 'top' },
            { path: '/static/sketch_parts/eyebrows/01.png', top: 180, scale: 0.58, layer: 'middle' },
            { path: '/static/sketch_parts/eyes/01.png', top: 215, scale: 0.58, layer: 'middle' },
            { path: '/static/sketch_parts/nose/01.png', top: 260, scale: 0.55, layer: 'middle' },
            { path: '/static/sketch_parts/lips/01.png', top: 345, scale: 0.55, layer: 'middle' }
        ];

        let loaded = 0;
        parts.forEach(p => {
            fabric.Image.fromURL(p.path, function (oImg) {
                if (oImg) {
                    oImg.scale(p.scale);
                    oImg.set({
                        originX: 'center',
                        originY: 'top',
                        left: canvas.getWidth() / 2,
                        top: p.top
                    });
                    canvas.add(oImg);
                    if (p.layer === 'bottom') oImg.sendToBack();
                    if (p.layer === 'top') oImg.bringToFront();
                }
                loaded++;
                if (loaded === parts.length) {
                    canvas.discardActiveObject();
                    canvas.renderAll();
                    updateLayerControls();
                }
            }, { crossOrigin: 'anonymous' });
        });
    });

    // --- 8. Clear Canvas ---
    document.getElementById('clearCanvasBtn')?.addEventListener('click', function () {
        if (confirm('Clear all canvas elements?')) {
            canvas.clear();
            canvas.backgroundColor = '#ffffff';
            centerGuideLine = null;
            if (isGuideVisible) createOrUpdateGuide();
            canvas.renderAll();
            updateLayerControls();
        }
    });

    // --- 9. Skin Tone Selection ---
    const skinToneInputs = document.querySelectorAll('input[name="selected_skin_tone"]');
    const skinToneInput = document.getElementById('skinToneInput');
    const skinLabel = document.getElementById('selectedSkinToneLabel');

    const toneDisplayNames = {
        'fair': 'Fair (Porcelain)',
        'light': 'Light (Peach)',
        'medium': 'Medium (Warm Beige)',
        'olive': 'Tan (Olive)',
        'brown': 'Deep Tan (Bronze)',
        'dark': 'Rich Dark (Espresso)'
    };

    skinToneInputs.forEach(radio => {
        radio.addEventListener('change', function () {
            if (this.checked) {
                const val = this.value;
                if (skinToneInput) skinToneInput.value = val;
                if (skinLabel) skinLabel.textContent = toneDisplayNames[val] || val;
            }
        });
    });

    // --- 10. Construct Face Dispatch with Loading State ---
    const constructFaceBtn = document.getElementById('constructFaceBtn');
    const loadingOverlay = document.getElementById('reconstruction-loading-overlay');
    const form = document.getElementById('constructFaceForm');
    const sketchDataInput = document.getElementById('sketchDataInput');

    constructFaceBtn?.addEventListener('click', function () {
        const objects = canvas.getObjects().filter(o => o !== centerGuideLine);
        if (objects.length === 0) {
            alert('Please place at least one facial feature on the canvas before constructing a face.');
            return;
        }

        // Hide guide line for clean sketch export
        if (centerGuideLine) centerGuideLine.set('visible', false);
        canvas.discardActiveObject();
        canvas.renderAll();

        // Export canvas PNG
        const dataURL = canvas.toDataURL({
            format: 'png',
            quality: 1.0,
            multiplier: 1.0
        });

        // Restore guide line visibility
        if (centerGuideLine && isGuideVisible) {
            centerGuideLine.set('visible', true);
            canvas.renderAll();
        }

        if (sketchDataInput && form) {
            sketchDataInput.value = dataURL;
            if (loadingOverlay) loadingOverlay.style.display = 'flex';
            constructFaceBtn.disabled = true;
            form.submit();
        }
    });

    // --- 11. Download Clean Sketch PNG ---
    document.getElementById('downloadBtn')?.addEventListener('click', function () {
        if (centerGuideLine) centerGuideLine.set('visible', false);
        canvas.discardActiveObject();
        canvas.renderAll();

        const dataURL = canvas.toDataURL({ format: 'png', quality: 1.0 });

        if (centerGuideLine && isGuideVisible) {
            centerGuideLine.set('visible', true);
            canvas.renderAll();
        }

        const link = document.createElement('a');
        link.download = 'thirdeye_composite_sketch.png';
        link.href = dataURL;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
