# Design Guidelines: TTS Text Editing Application

## Design Approach
**System-Based Approach**: Modern developer tool aesthetic inspired by Linear, VSCode, and Notion - emphasizing clarity, efficiency, and information density for a technical audience.

## Core Design Principles
- **Clarity First**: Every UI element serves a clear functional purpose
- **Information Hierarchy**: Guide users through complex workflows with visual structure
- **Technical Elegance**: Professional, tool-focused design without unnecessary decoration
- **Responsive Efficiency**: Optimize for both focused work and quick scanning

## Color Palette

### Dark Mode (Primary)
- **Background Base**: 220 15% 10% (primary canvas)
- **Background Elevated**: 220 15% 14% (cards, panels)
- **Background Interactive**: 220 15% 18% (hover states)
- **Border Subtle**: 220 10% 20% (dividers)
- **Border Interactive**: 220 15% 30% (focus states)
- **Text Primary**: 220 10% 95%
- **Text Secondary**: 220 8% 70%
- **Text Tertiary**: 220 5% 50%

### Accent Colors
- **Primary Action**: 212 100% 48% (buttons, links, active states)
- **Success**: 142 76% 45% (completed chunks, validation success)
- **Warning**: 38 92% 50% (retry indicators)
- **Error**: 0 72% 51% (failed validation, errors)
- **Info**: 199 89% 48% (progress indicators)

### Light Mode Support
- **Background Base**: 220 15% 98%
- **Background Elevated**: 0 0% 100%
- **Text Primary**: 220 15% 15%
- **Border Subtle**: 220 10% 88%

## Typography

### Font Stack
- **Primary**: 'Inter', system-ui, -apple-system, sans-serif (UI elements, body text)
- **Monospace**: 'JetBrains Mono', 'Fira Code', monospace (code, logs, technical output)

### Type Scale
- **Display**: text-3xl font-semibold (page headers)
- **Heading**: text-xl font-semibold (section titles)
- **Subheading**: text-base font-medium (component labels)
- **Body**: text-sm (primary content, form labels)
- **Caption**: text-xs (helper text, metadata)
- **Code**: text-sm font-mono (logs, technical output)

## Layout System

### Spacing Primitives
Use Tailwind units: **2, 4, 6, 8, 12, 16** for consistent rhythm
- Component padding: p-4 to p-6
- Section gaps: gap-6 to gap-8
- Page margins: p-6 to p-8
- Icon-text spacing: gap-2

### Grid Structure
- **Main Container**: max-w-7xl mx-auto px-6
- **Two-Column Layouts**: grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6
- **Control Panels**: Single column forms with full-width inputs

## Component Library

### Primary Components

**File Upload Zone**
- Dashed border, background elevated color
- Large dropzone area with icon and supporting text
- File preview card with metadata (word count, character count)
- Clear/replace actions

**Configuration Panel**
- Grouped checkbox controls with clear labels
- Collapsible sections for advanced options
- Number inputs for batch size configuration
- Radio group for mode selection (Mode 1: Format Conversion / Mode 2: Intelligent Parsing)

**Processing Interface**
- Full-width progress bar with percentage and chunk status
- Live updating text output in monospace font with syntax highlighting for speaker labels
- Real-time log panel with timestamp, action type, and status icons

**Action Buttons**
- Primary: Solid background with primary color
- Secondary: Outline variant with subtle background
- Icon buttons: Square with rounded corners, icon-only for compact actions

### Data Display Components

**Log Panel**
- Table-like structure with timestamp, event type, and message columns
- Color-coded status indicators (success: green, warning: orange, error: red)
- Scrollable container with fixed height
- Export/clear actions in panel header

**Output Text Area**
- Monospace font for accurate character representation
- Speaker label syntax highlighting (different color for [1]:, Speaker 2:, etc.)
- Line numbers in gutter (optional)
- Toolbar with copy and save actions

**Speaker Configuration**
- Dynamic input fields based on speaker count
- Name mapping interface for Mode 2 (detected name → speaker label)
- Visual speaker color coding for preview

### Forms & Inputs

**Text Inputs**
- Consistent height (h-10), rounded corners (rounded-md)
- Border with border-subtle color, focus ring with primary color
- Label above input with text-sm font-medium
- Helper text below in text-xs text-tertiary

**Checkboxes**
- Custom styled with primary accent color when checked
- Clear label with optional description text
- Grouped in vertical list with consistent spacing

**Select/Dropdown**
- Native select styled to match design system
- Chevron icon indicator
- Same height as text inputs

## Page Layout

### Main Application View

**Header** (sticky top, h-16)
- Logo/app name on left
- Theme toggle on right
- Subtle bottom border

**Three-Column Layout** (on desktop)
1. **Left Panel** (w-80): Configuration controls, file upload, processing options
2. **Center Panel** (flex-1): Output text area with live updates
3. **Right Panel** (w-96): Activity log, progress indicator

**Mobile Stacking** (single column)
- Configuration → Output → Log (vertical flow)
- Fixed bottom action bar for primary controls

### Information Architecture

**Processing Workflow Sections:**
1. File Input & Preview
2. Text Repair Configuration
3. Multi-Speaker Settings
4. Processing Control (Start/Stop buttons)
5. Live Output Display
6. Activity Log & Export

## Interaction Patterns

**Processing States**
- Idle: Default state with all controls enabled
- Processing: Progress animation, disable edit controls, show cancel option
- Complete: Success state, enable export actions, show summary stats

**Real-time Updates**
- Chunk-by-chunk text streaming to output area
- Log entries append with smooth animation
- Progress bar fills with easing transition

**Error Handling**
- Inline validation messages below inputs
- Toast notifications for system-level errors
- Detailed error logs in activity panel

## Accessibility

- ARIA labels for all interactive elements
- Keyboard navigation for all workflows
- Focus indicators visible in both light and dark modes
- Screen reader announcements for processing status changes
- Sufficient color contrast (WCAG AA minimum)

## Performance Considerations

- Virtualized scrolling for large log outputs
- Debounced text area updates for smooth rendering
- Lazy loading for configuration panels
- Optimistic UI updates with rollback on errors