import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { StudioLayout } from "./components/layout/StudioLayout";
import { CommandCenterPage } from "./pages/CommandCenterPage";
import { CompareLabPage } from "./pages/CompareLabPage";
import { PaperStudioPage } from "./pages/PaperStudioPage";
import { ReceiptVaultPage } from "./pages/ReceiptVaultPage";
import { RunLauncherPage } from "./pages/RunLauncherPage";
import { SettingsPage } from "./pages/SettingsPage";
import { TestWorkbenchPage } from "./pages/TestWorkbenchPage";

export function App() {
  return (
    <BrowserRouter>
      <StudioLayout>
        <Routes>
          <Route path="/" element={<CommandCenterPage />} />
          <Route path="/compare" element={<CompareLabPage />} />
          <Route path="/run" element={<RunLauncherPage />} />
          <Route path="/papers" element={<PaperStudioPage />} />
          <Route path="/tests" element={<TestWorkbenchPage />} />
          <Route path="/receipts" element={<ReceiptVaultPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </StudioLayout>
    </BrowserRouter>
  );
}

export default App;
