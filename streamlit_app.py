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

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers_mask = iso_forest.fit_predict(data_scaled) == -1

        outliers = pd.Series(False, index=data.index)
        outliers.loc[data.dropna().index] = outliers_mask

        return outliers

    @staticmethod
    def detect_consecutive_duplicates(data, min_consecutive=3):
        """Detect consecutive duplicate values"""
        outliers = pd.Series(False, index=data.index)

        for i in range(len(data) - min_consecutive + 1):
            window = data.iloc[i:i + min_consecutive]
            if len(window.unique()) == 1 and not pd.isna(window.iloc[0]):
                outliers.iloc[i:i + min_consecutive] = True

        return outliers

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
            if st.session_state.original_data is None:
                st.session_state.original_data = df.copy()
            st.session_state.data = df

            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

            # Data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(), use_container_width=True)

            # Column selection
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_columns:
                st.error("No numeric columns found for anomaly detection")
                return

            col1, col2 = st.columns([2, 1])

            with col1:
                selected_columns = st.multiselect(
                    "Select columns for anomaly detection",
                    numeric_columns,
                    default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
                )

            with col2:
                auto_detect = st.button(
                    "🔍 Auto-Detect Anomalies",
                    type="primary",
                    use_container_width=True
                )

            if selected_columns and auto_detect:
                detect_anomalies(df, selected_columns)

            # Display detected anomalies
            if st.session_state.detected_anomalies:
                display_anomaly_results(df, selected_columns)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def detect_anomalies(df, columns):
    """Detect anomalies in selected columns"""
    detector = AnomalyDetector()

    with st.spinner("🔍 Detecting anomalies..."):
        anomalies = {}

        for column in columns:
            data = df[column]
            column_anomalies = {}

            # Statistical outliers (IQR method)
            iqr_outliers = detector.detect_statistical_outliers(data, method='iqr')
            if iqr_outliers.sum() > 0:
                column_anomalies['IQR Outliers'] = {
                    'indices': iqr_outliers[iqr_outliers].index.tolist(),
                    'confidence': 'High',
                    'description': f'Values outside 1.5×IQR range'
                }

            # Z-score outliers
            zscore_outliers = detector.detect_statistical_outliers(data, method='zscore', threshold=3)
            if zscore_outliers.sum() > 0:
                column_anomalies['Z-Score Outliers'] = {
                    'indices': zscore_outliers[zscore_outliers].index.tolist(),
                    'confidence': 'High',
                    'description': f'Z-score > 3 standard deviations'
                }

            # Isolation Forest
            if len(data.dropna()) >= 10:
                iso_outliers = detector.detect_isolation_forest(data)
                if iso_outliers.sum() > 0:
                    column_anomalies['Isolation Forest'] = {
                        'indices': iso_outliers[iso_outliers].index.tolist(),
                        'confidence': 'Medium',
                        'description': 'Machine learning detected anomalies'
                    }

            # Consecutive duplicates
            consecutive_dups = detector.detect_consecutive_duplicates(data)
            if consecutive_dups.sum() > 0:
                column_anomalies['Consecutive Duplicates'] = {
                    'indices': consecutive_dups[consecutive_dups].index.tolist(),
                    'confidence': 'Medium',
                    'description': '3+ consecutive identical values'
                }

            if column_anomalies:
                anomalies[column] = column_anomalies

        st.session_state.detected_anomalies = anomalies

        if anomalies:
            st.success(f"🎯 Found anomalies in {len(anomalies)} columns")
        else:
            st.info("✨ No significant anomalies detected")

def display_anomaly_results(df, selected_columns):
    """Display anomaly detection results with interactive charts"""

    st.subheader("🎯 Detected Anomalies")

    for column, methods in st.session_state.detected_anomalies.items():
        with st.expander(f"📈 {column} - {len(methods)} anomaly types detected", expanded=True):

            # Create visualization
            fig = create_anomaly_chart(df, column, methods)
            st.plotly_chart(fig, use_container_width=True)

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

                # Confirmation button
                result = confirmation_button(
                    key=f"confirm_{column}",
                    width="content"
                )

                if result and result.get("confirmed"):
                    remove_all_anomalies(column)
                    st.rerun()

def create_anomaly_chart(df, column, methods):
    """Create interactive chart showing anomalies"""
    fig = go.Figure()

    # Plot original data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode='lines+markers',
        name='Original Data',
        line=dict(color='blue', width=1),
        marker=dict(size=4)
    ))

    # Highlight anomalies by method
    colors = ['red', 'orange', 'purple', 'brown']

    for i, (method, details) in enumerate(methods.items()):
        indices = details['indices']
        fig.add_trace(go.Scatter(
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
    """Automatically remove high confidence anomalies"""
    # Save current state for undo
    st.session_state.undo_stack.append(st.session_state.data.copy())

    indices_to_remove = set()
    for method in methods:
        if method in st.session_state.detected_anomalies[column]:
            indices_to_remove.update(st.session_state.detected_anomalies[column][method]['indices'])

    # Remove anomalies by setting to NaN
    st.session_state.data.loc[list(indices_to_remove), column] = np.nan

    # Update detected anomalies
    if column in st.session_state.detected_anomalies:
        del st.session_state.detected_anomalies[column]

    st.success(f"✅ Automatically removed {len(indices_to_remove)} high-confidence anomalies from {column}")

def remove_all_anomalies(column):
    """Remove all detected anomalies for a column"""
    # Save current state for undo
    st.session_state.undo_stack.append(st.session_state.data.copy())

    all_indices = set()
    for details in st.session_state.detected_anomalies[column].values():
        all_indices.update(details['indices'])

    # Remove anomalies by setting to NaN
    st.session_state.data.loc[list(all_indices), column] = np.nan

    # Update detected anomalies
    if column in st.session_state.detected_anomalies:
        del st.session_state.detected_anomalies[column]

    st.success(f"✅ Removed {len(all_indices)} anomalies from {column}")

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
        current_rows = len(st.session_state.data.dropna())
        removed_rows = original_rows - current_rows

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
                title=f"Cleaned {selected_column}"
            )
            fig_cleaned.update_layout(height=300)
            st.plotly_chart(fig_cleaned, use_container_width=True)

            # Cleaned statistics
            cleaned_stats = st.session_state.data[selected_column].describe()
            st.write("**Statistics:**")
            st.write(f"Mean: {cleaned_stats['mean']:.2f}")
            st.write(f"Std: {cleaned_stats['std']:.2f}")
            st.write(f"Min: {cleaned_stats['min']:.2f}")
            st.write(f"Max: {cleaned_stats['max']:.2f}")

def display_advanced_settings():
    """Display advanced anomaly detection settings"""
    with st.sidebar:
        st.header("⚙️ Advanced Settings")

        st.subheader("Detection Thresholds")

        iqr_threshold = st.slider(
            "IQR Multiplier",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Higher values = less sensitive to outliers"
        )

        zscore_threshold = st.slider(
            "Z-Score Threshold",
            min_value=2.0,
            max_value=4.0,
            value=3.0,
            step=0.1,
            help="Higher values = less sensitive to outliers"
        )

        isolation_contamination = st.slider(
            "Isolation Forest Contamination",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Expected proportion of outliers"
        )

        consecutive_threshold = st.slider(
            "Consecutive Duplicates Threshold",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            help="Minimum consecutive identical values to flag"
        )

        st.subheader("Auto-removal Settings")

        auto_remove_high_confidence = st.checkbox(
            "Auto-remove high confidence anomalies",
            value=False,
            help="Automatically remove anomalies with high confidence without confirmation"
        )

        return {
            'iqr_threshold': iqr_threshold,
            'zscore_threshold': zscore_threshold,
            'isolation_contamination': isolation_contamination,
            'consecutive_threshold': consecutive_threshold,
            'auto_remove_high_confidence': auto_remove_high_confidence
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
        removed_rows = original_shape[0] - current_shape[0]
        st.metric("Rows Removed", removed_rows)
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
    # Display advanced settings in sidebar
    settings = display_advanced_settings()

    # Run main application
    main()

    # Display comparison charts if data is available
    if st.session_state.data is not None and st.session_state.original_data is not None:
        display_comparison_charts()
        create_summary_report()

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
