import Chat from "./components/Chat";

export default function Page() {
  return (
    <main className="shell">
      <header className="header">
        <h1 className="title">WikiBrain</h1>
        <p className="subtitle">
          Answers drawn only from company policy documents — every answer shows
          its source.
        </p>
      </header>
      <Chat />
    </main>
  );
}
