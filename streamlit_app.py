import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import io

# Configure page
st.set_page_config(
    page_title="Automated Data Cleansing Tool",
    page_icon="🧹",
    layout="wide"
)

# Custom component for confirmation button
confirmation_button = st.components.v2.component(
    name="confirmation_button",
    html="""
    <button class="confirm-button" id="confirm-btn">
        <div class="button-content">
            <span class="icon" id="icon">🗑️</span>
            <span class="label" id="label">Hold to Confirm</span>
        </div>
        <div class="progress-bar" id="progress"></div>
    </button>
    """,
    css="""
    .confirm-button {
        position: relative;
        padding: 12px 24px;
        border: 2px solid #ff4444;
        background: #fff;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        overflow: hidden;
    }

    .confirm-button:hover {
        background: #fff5f5;
        transform: scale(1.02);
    }

    .confirm-button.holding {
        background: #ffe6e6;
        border-color: #ff0000;
    }

    .button-content {
        display: flex;
        align-items: center;
        gap: 8px;
        position: relative;
        z-index: 2;
    }

    .progress-bar {
        position: absolute;
        bottom: 0;
        left: 0;
        height: 4px;
        background: #ff4444;
        width: 0%;
        transition: width 0.1s linear;
    }

    .icon { font-size: 1.2rem; }
    .label { 
        font-weight: 600;
        color: #333;
    }
    """,
    js="""
    const HOLD_DURATION = 2000;

    export default function({ parentElement, setTriggerValue }) {
        const button = parentElement.querySelector("#confirm-btn");
        const progress = parentElement.querySelector("#progress");
        const label = parentElement.querySelector("#label");

        let startTime = null;
        let animationFrame = null;

        function updateProgress() {
            if (!startTime) return;

            const elapsed = Date.now() - startTime;
            const progressPercent = Math.min(elapsed / HOLD_DURATION, 1);

            progress.style.width = (progressPercent * 100) + "%";

            if (progressPercent >= 1) {
                triggerAction();
            } else {
                animationFrame = requestAnimationFrame(updateProgress);
            }
        }

        function startHold() {
            startTime = Date.now();
            button.classList.add("holding");
            label.textContent = "Keep holding...";
            animationFrame = requestAnimationFrame(updateProgress);
        }

        function cancelHold() {
            startTime = null;
            button.classList.remove("holding");
            label.textContent = "Hold to Confirm";
            progress.style.width = "0%";

            if (animationFrame) {
                cancelAnimationFrame(animationFrame);
                animationFrame = null;
            }
        }

        function triggerAction() {
            cancelAnimationFrame(animationFrame);
            setTriggerValue("confirmed", true);

            label.textContent = "Confirmed!";
            setTimeout(() => {
                label.textContent = "Hold to Confirm";
                progress.style.width = "0%";
                button.classList.remove("holding");
            }, 1000);
        }

        button.addEventListener("mousedown", startHold);
        button.addEventListener("mouseup", cancelHold);
        button.addEventListener("mouseleave", cancelHold);

        return () => {
            if (animationFrame) cancelAnimationFrame(animationFrame);
            button.removeEventListener("mousedown", startHold);
            button.removeEventListener("mouseup", cancelHold);
            button.removeEventListener("mouseleave", cancelHold);
        };
    }
    """
)

class AnomalyDetector:
    """Automated anomaly detection using multiple methods"""

    @staticmethod
    def detect_statistical_outliers(data, method='iqr', threshold=1.5):
        """Detect outliers using statistical methods"""
        outliers = pd.Series(False, index=data.index)

        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data.dropna()))
            outliers.loc[data.dropna().index] = z_scores > threshold

        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold

        return outliers

    @staticmethod
    def detect_isolation_forest(data, contamination=0.1):
        """Detect outliers using Isolation Forest"""
        if len(data.dropna()) < 10:
            return pd.Series(False, index=data.index)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.dropna().values.reshape(-1, 1))

        iso_forest = IsolationForest(contamination=contamination, n_estimators=100, random_state=42, n_jobs=-1)

        outliers_mask = iso_forest.fit_predict(data_scaled) == -1

        outliers = pd.Series(False, index=data.index)
        outliers.loc[data.dropna().index] = outliers_mask

        return outliers
    
    @staticmethod
    def detect_voltage_anomalies(
        data,
        window=15,
        residual_z_threshold=4,
        gradient_z_threshold=4
    ):

        outliers = pd.Series(False, index=data.index)

        s = pd.to_numeric(data, errors="coerce")
        valid = s.dropna()

        if len(valid) < window:
            return outliers

        trend = valid.rolling(
            window=window,
            center=True,
            min_periods=5
        ).median()

        residual = valid - trend

        residual_median = np.median(residual)
        residual_mad = np.median(
            np.abs(residual - residual_median)
        )

        residual_mad = max(residual_mad, 1e-9)

        residual_z = (
            0.6745 *
            (residual - residual_median)
            / residual_mad
        )

        gradient = valid.diff()

        gradient_median = np.median(
            gradient.dropna()
        )

        gradient_mad = np.median(
            np.abs(
                gradient.dropna()
                - gradient_median
            )
        )

        gradient_mad = max(
            gradient_mad,
            1e-9
        )

        gradient_z = (
            0.6745 *
            (gradient - gradient_median)
            / gradient_mad
        )

        spike_mask = (
            np.abs(residual_z)
            > residual_z_threshold
        ) & (
            np.abs(gradient_z)
            > gradient_z_threshold
        )

        prev_val = valid.shift(1)
        next_val = valid.shift(-1)

        local_mid = (
            prev_val + next_val
        ) / 2

        deviation = np.abs(
            valid - local_mid
        )

        baseline_jump = np.median(
            np.abs(valid.diff())
        )

        reversal_spikes = (
            deviation >
            baseline_jump * 6
        )

        before = (
            valid
            .rolling(10, min_periods=3)
            .mean()
            .shift(1)
        )

        after = (
            valid[::-1]
            .rolling(10, min_periods=3)
            .mean()[::-1]
        )

        step_shift = np.abs(
            after - before
        )

        step_mask = (
            step_shift >
            baseline_jump * 8
        )

        score = (
            spike_mask.astype(int) * 3 +
            reversal_spikes.astype(int) * 4 +
            step_mask.astype(int) * 3
        )

        final_mask = score >= 4

        outliers.loc[
            final_mask[final_mask].index
        ] = True

        return outliers

    
    @staticmethod
    def detect_consecutive_duplicates(
        data,
        min_consecutive=3
    ):

        groups = (
            data != data.shift()
        ).cumsum()

        counts = (
            data.groupby(groups)
            .transform("size")
        )

        return (
            counts >= min_consecutive
        ) & data.notna()

    @staticmethod
    def detect_zero_values(data):
        """Detect zero values"""
        return (pd.to_numeric(data, errors="coerce") == 0)  

def main():
    st.title("🧹 Automated Data Cleansing Tool")
    st.caption("Upload your data and let AI help you identify and remove anomalies")

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'detected_anomalies' not in st.session_state:
        st.session_state.detected_anomalies = {}
    if 'pending_removals' not in st.session_state:
        st.session_state.pending_removals = {}
    if 'undo_stack' not in st.session_state:
        st.session_state.undo_stack = []
    if 'manual_selections' not in st.session_state:
        st.session_state.manual_selections = {}
    if 'auto_detect_done' not in st.session_state:
        st.session_state.auto_detect_done = False
    if 'last_selection' not in st.session_state:
        st.session_state.last_selection = ()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your CSV file for automated anomaly detection"
    )

    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            df = df.replace(0, np.nan)

            if st.session_state.original_data is None or uploaded_file.name != st.session_state.get("last_file"):
                st.session_state.detected_anomalies = {}
                st.session_state.manual_selections = {}
                st.session_state.last_selection = ()
                st.session_state.auto_detect_done = False
                st.session_state.original_data = df.copy()
                st.session_state.data = df.copy()
                st.session_state.last_file = uploaded_file.name

            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

            st.info(f"Dataset contains {len(df):,} rows")
            # Data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(), use_container_width=True)

            # Column selection
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_columns:
                st.error("No numeric columns found for anomaly detection")
                return

            selected_columns = st.multiselect(
                "Select columns for anomaly detection",
                numeric_columns,
                default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
            )

            current_selection = tuple(sorted(selected_columns))

            if (
                st.session_state.get("last_selection")
                != current_selection
            ):

                detect_anomalies(
                    st.session_state.data,
                    selected_columns,
                    settings
                )

                st.session_state.last_selection = (
                    current_selection
                )

            if selected_columns:

                if st.button(
                    "🔍 Detect Anomalies",
                    key="run_detection"
                ):
                    detect_anomalies(
                        st.session_state.data,
                        selected_columns,
                        settings
                    )

            # Display detected anomalies
            if st.session_state.detected_anomalies:
                display_anomaly_results(st.session_state.data, selected_columns)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def detect_anomalies(df, columns, settings):
    """Detect anomalies in selected columns"""
    detector = AnomalyDetector()

    with st.spinner("🔍 Detecting anomalies..."):
        anomalies = {}

        for column in columns:
            data = df[column]
            column_anomalies = {}

            # Statistical outliers (IQR method)
            iqr_outliers = detector.detect_statistical_outliers(
                data,
                method='iqr',
                threshold=settings['iqr_threshold']
            )
            if iqr_outliers.sum() > 0:
                column_anomalies['IQR Outliers'] = {
                    'indices': iqr_outliers[iqr_outliers].index.tolist(),
                    'confidence': 'High',
                    'description': f'Values outside {settings["iqr_threshold"]}×IQR range'
                }

            # Z-score outliers
            zscore_outliers = detector.detect_statistical_outliers(
                data,
                method='zscore',
                threshold=settings['zscore_threshold']
            )
            if zscore_outliers.sum() > 0:
                column_anomalies['Z-Score Outliers'] = {
                    'indices': zscore_outliers[zscore_outliers].index.tolist(),
                    'confidence': 'High',
                    'description': f'Z-score > {settings["zscore_threshold"]} standard deviations'
                }

            # Sudden spikes
            voltage_outliers = ( 
                detector.detect_voltage_anomalies(
                    data,
                    window=5,
                )
            )
            
            if voltage_outliers.sum() > 0:
                column_anomalies['Voltage Anomalies'] = {
                    'indices': voltage_outliers[
                        voltage_outliers
                    ].index.tolist(),
                    'confidence': 'High',
                    'description':
                        'Detected spikes, drops, trend breaks and step changes'
                }

            # Isolation Forest
            if len(data.dropna()) >= 10:
                iso_outliers = detector.detect_isolation_forest(
                    data,
                    contamination=settings['isolation_contamination']
                )
                if iso_outliers.sum() > 0:
                    column_anomalies['Isolation Forest'] = {
                        'indices': iso_outliers[iso_outliers].index.tolist(),
                        'confidence': 'Medium',
                        'description': 'Machine learning detected anomalies'
                    }

            # Consecutive duplicates
            consecutive_dups = detector.detect_consecutive_duplicates(
                data,
                min_consecutive=settings['consecutive_threshold']
            )
            if consecutive_dups.sum() > 0:
                column_anomalies['Consecutive Duplicates'] = {
                    'indices': consecutive_dups[consecutive_dups].index.tolist(),
                    'confidence': 'Medium',
                    'description': f'{settings["consecutive_threshold"]}+ consecutive identical values'
                }

            if column_anomalies:
                anomalies[column] = column_anomalies

        st.session_state.detected_anomalies = anomalies

        if anomalies:
            st.success(f"🎯 Found anomalies in {len(anomalies)} columns")
        else:
            st.info("✨ No significant anomalies detected")

    # Zero values
    zero_outliers = detector.detect_zero_values(data)

    if zero_outliers.sum() > 0:
        column_anomalies['Zero Values'] = {
            'indices': zero_outliers[zero_outliers].index.tolist(),
            'confidence': 'High',
            'description': 'Zero values detected'
        }

def display_anomaly_results(df, selected_columns):
    """Display anomaly detection results with interactive charts"""

    st.subheader("🎯 Detected Anomalies")

    # Use a snapshot so actions that delete entries from session_state do not
    # mutate the dictionary while it is being iterated.
    for column, methods in list(st.session_state.detected_anomalies.items()):
        with st.expander(f"📈 {column} - {len(methods)} anomaly types detected", expanded=True):

            # Create visualization
            fig = create_anomaly_chart(df, column, methods)
            chart_event = st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"chart_{column}",
                on_select="rerun",
                selection_mode=['points', 'box']
            )

            selected_indices = st.session_state.manual_selections.get(column,[])

            if chart_event:

                selection_state = (
                    chart_event.get("selection", {})
                    if isinstance(chart_event, dict)
                    else getattr(chart_event, "selection", {})
                )

                if selection_state:

                    selected_indices = [
                        int(i)
                        for i in selection_state.get(
                            "point_indices",
                            []
                        )
                    ]

                    if selected_indices:
                        st.session_state.manual_selections[column] = (
                            sorted(set(selected_indices))
                        )
                    else:
                        selected_indices = (
                            st.session_state.manual_selections.get(
                                column,
                                []
                            )
                        )

            if selected_indices:
                st.session_state.manual_selections[column] = sorted(set(selected_indices))
                st.caption("🖱️ Drag a box on the chart or select points, then remove the highlighted region.")

            # Anomaly details and actions
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Detected Anomaly Types:**")

                for method, details in methods.items():
                    confidence_color = "🔴" if details['confidence'] == 'High' else "🟡"
                    st.write(f"{confidence_color} **{method}** ({details['confidence']} confidence)")
                    st.write(f"   • {details['description']}")
                    st.write(f"   • {len(details['indices'])} data points affected")

                    # Show sample values
                    sample_indices = details['indices'][:5]
                    sample_values = df.loc[sample_indices, column].tolist()
                    st.write(f"   • Sample values: {sample_values}")
                    st.write("")

            with col2:
                st.write("**Actions:**")

                # Auto-remove high confidence anomalies
                high_confidence_methods = [m for m, d in methods.items() if d['confidence'] == 'High']

                if high_confidence_methods:
                    if st.button(f"🤖 Auto-remove High Confidence", key=f"auto_{column}"):
                        auto_remove_anomalies(column, high_confidence_methods)

                # Manual confirmation for all anomalies
                st.write("**Manual Review Required:**")
                all_indices = set()
                for details in methods.values():
                    all_indices.update(details['indices'])

                st.write(f"Total anomalies: {len(all_indices)}")

                manual_selected = st.session_state.manual_selections.get(column, [])
                if manual_selected:
                    st.info(f"Selected {len(manual_selected)} chart points for manual removal.")
                    if st.button(f"🗑️ Remove Selected Region", key=f"manual_remove_{column}"):
                        remove_selected_region(column, manual_selected)

                # Confirmation button
                result = confirmation_button(
                    key=f"confirm_{column}",
                    width="content"
                )

                if result and result.get("confirmed"):
                    remove_all_anomalies(column)

def create_anomaly_chart(df, column, methods):
    """Create interactive chart showing anomalies"""

    plot_df = df

    fig = go.Figure()

    # Plot original data
    fig.add_trace(go.Scattergl(
        x=plot_df.index,
        y=plot_df[column],
        mode='lines+markers',
        name='Current Data',
        connectgaps=False,
        line=dict(color='blue', width=1),
        marker=dict(size=2)
    ))

    removed_mask = df[column].isna()

    if removed_mask.any():

        fig.add_trace(
            go.Scattergl(
                x=df.index[removed_mask],
                y=st.session_state.original_data.loc[
                    removed_mask,
                    column
                ],
                mode="markers",
                marker=dict(
                    color="black",
                    size=10,
                    symbol="x"
                ),
                name="Removed Values"
            )
        )


    # Highlight anomalies by method
    colors = ['red', 'orange', 'purple', 'brown']

    for i, (method, details) in enumerate(methods.items()):
        indices = details['indices']

        fig.add_trace(go.Scattergl(
            x=indices,
            y=df.loc[indices, column],
            mode='markers',
            name=f'{method} ({len(indices)} points)',
            marker=dict(
                color=colors[i % len(colors)],
                size=8,
                symbol='x'
            )
        ))

    fig.update_layout(
        title=f'Anomaly Detection Results for {column}',
        xaxis_title='Index',
        yaxis_title=column,
        hovermode='closest',
        height=400
    )

    return fig

def auto_remove_anomalies(column, methods):

    st.session_state.undo_stack.append(
        st.session_state.data.copy()
    )

    indices_to_remove = set()

    for method in methods:
        if (
            column in st.session_state.detected_anomalies
            and method in st.session_state.detected_anomalies[column]
        ):
            indices_to_remove.update(
                st.session_state.detected_anomalies[column][method]["indices"]
            )

    st.session_state.data.loc[
        list(indices_to_remove),
        column
    ] = np.nan

    st.rerun()
def remove_selected_region(column, selected_indices):

    if not selected_indices:
        return

    st.session_state.undo_stack.append(
        st.session_state.data.copy()
    )

    selected_rows = (
        st.session_state.data
        .index[selected_indices]
        .tolist()
    )

    st.session_state.data.loc[
        selected_rows,
        column
    ] = np.nan

    st.rerun()

def remove_all_anomalies(column):

    st.session_state.undo_stack.append(
        st.session_state.data.copy()
    )

    all_indices = set() 

    for details in st.session_state.detected_anomalies[column].values():
        all_indices.update(details["indices"])

    st.session_state.data.loc[
        list(all_indices),
        column
    ] = np.nan

    st.rerun()

def undo_last_action():
    """Undo the last data modification"""
    if st.session_state.undo_stack:
        st.session_state.data = st.session_state.undo_stack.pop()
        st.success("↩️ Last action undone")

def display_download_section():
    """Display download options for cleaned data"""
    st.subheader("📥 Download Cleaned Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Statistics
        original_rows = len(st.session_state.original_data) if st.session_state.original_data is not None else 0
        
        removed_rows = (
            st.session_state.data
            .isna()
            .sum()
            .sum()
        )

        current_rows = len(st.session_state.data)
        st.metric("Original Rows", original_rows)
        st.metric("Cleaned Rows", current_rows)
        st.metric("Removed Anomalies", removed_rows)

    with col2:
        # Data quality metrics
        if st.session_state.data is not None:
            missing_percentage = (st.session_state.data.isnull().sum().sum() / 
                                (len(st.session_state.data) * len(st.session_state.data.columns))) * 100

            st.metric("Missing Data %", f"{missing_percentage:.1f}%")
            st.metric("Columns", len(st.session_state.data.columns))

    with col3:
        # Download buttons
        if st.session_state.data is not None:
            # Create CSV download
            csv_buffer = io.StringIO()
            st.session_state.data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Create Excel download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                st.session_state.data.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                if st.session_state.original_data is not None:
                    st.session_state.original_data.to_excel(writer, sheet_name='Original_Data', index=False)

            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

def display_comparison_charts():
    """Display before/after comparison charts"""
    if st.session_state.original_data is None or st.session_state.data is None:
        return

    st.subheader("📊 Before vs After Comparison")

    numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        st.info("No numeric columns available for comparison")
        return

    selected_column = st.selectbox(
        "Select column for comparison",
        numeric_columns,
        key="comparison_column"
    )

    if selected_column:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Data**")
            fig_original = px.line(
                x=st.session_state.original_data.index,
                y=st.session_state.original_data[selected_column],
                title=f"Original {selected_column}"
            )

            fig_original.update_traces(
                connectgaps=False
            )

            fig_original.update_layout(height=300)
            st.plotly_chart(fig_original, use_container_width=True)

            # Original statistics
            orig_stats = st.session_state.original_data[selected_column].describe()
            st.write("**Statistics:**")
            st.write(f"Mean: {orig_stats['mean']:.2f}")
            st.write(f"Std: {orig_stats['std']:.2f}")
            st.write(f"Min: {orig_stats['min']:.2f}")
            st.write(f"Max: {orig_stats['max']:.2f}")

        with col2:
            st.write("**Cleaned Data**")
            fig_cleaned = px.line(
                x=st.session_state.data.index,
                y=st.session_state.data[selected_column],
                markers=True,
                title=f"Cleaned {selected_column}"
            )

            fig_cleaned.update_traces(
                connectgaps=False
            )

            fig_cleaned.update_layout(
                height=300
            )

            st.plotly_chart(fig_cleaned, use_container_width=True)

            # Cleaned statistics
            cleaned_stats = st.session_state.data[selected_column].describe()
            st.write("**Statistics:**")
            st.write(f"Mean: {cleaned_stats['mean']:.2f}")
            st.write(f"Std: {cleaned_stats['std']:.2f}")
            st.write(f"Min: {cleaned_stats['min']:.2f}")
            st.write(f"Max: {cleaned_stats['max']:.2f}")

def get_default_detection_settings():
    """Return built-in automatic anomaly-detection settings."""
    return {
        'iqr_threshold': 1.5,
        'zscore_threshold': 3.0,
        'spike_threshold': 6.0,
        'isolation_contamination': 0.1,
        'consecutive_threshold': 3,
        'auto_remove_high_confidence': False
    }

def create_summary_report():
    """Create a summary report of the cleaning process"""
    if st.session_state.original_data is None or st.session_state.data is None:
        return

    st.subheader("📋 Cleaning Summary Report")

    # Overall statistics
    original_shape = st.session_state.original_data.shape
    current_shape = st.session_state.data.shape

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Original Rows", original_shape[0])
    with col2:
        st.metric("Current Rows", current_shape[0])
    with col3:

        removed_rows = (
            st.session_state.data
            .isna()
            .sum()
            .sum()
        )
        
        st.metric("Values Removed", removed_rows)
    with col4:
        removal_percentage = (removed_rows / original_shape[0]) * 100 if original_shape[0] > 0 else 0
        st.metric("Removal %", f"{removal_percentage:.1f}%")

    # Column-wise analysis
    st.write("**Column-wise Changes:**")

    numeric_columns = st.session_state.original_data.select_dtypes(include=[np.number]).columns

    summary_data = []
    for column in numeric_columns:
        original_count = st.session_state.original_data[column].count()
        current_count = st.session_state.data[column].count()
        removed_count = original_count - current_count

        summary_data.append({
            'Column': column,
            'Original Count': original_count,
            'Current Count': current_count,
            'Removed': removed_count,
            'Removal %': f"{(removed_count / original_count * 100):.1f}%" if original_count > 0 else "0%"
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

# Main execution
if __name__ == "__main__":
    # Use automatic built-in detection thresholds
    settings = get_default_detection_settings()

    # Run main application
    main()

    if st.session_state.data is not None and st.session_state.original_data is not None:

        display_comparison_charts()
        create_summary_report()
        display_download_section()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **💡 Tips for effective data cleaning:**
        - Start with high-confidence anomalies (IQR and Z-score methods)
        - Review medium-confidence anomalies manually
        - Use the comparison charts to validate your cleaning decisions
        - Keep backups of your original data
        - Document your cleaning process for reproducibility
        """
    )
