import { useEffect, useRef, type FormEvent, type KeyboardEvent } from "react";

type ComposerProps = {
  value: string;
  busy: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
};

export function Composer({
  value,
  busy,
  onChange,
  onSubmit,
  onStop,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  }, [value]);

  useEffect(() => {
    if (!busy) textareaRef.current?.focus();
  }, [busy]);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!busy && value.trim()) onSubmit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      if (!busy && value.trim()) onSubmit();
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={textareaRef}
        aria-label="Message the agent"
        placeholder="Message the agent"
        rows={1}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      {busy ? (
        <button
          className="composer-action stop-action"
          type="button"
          onClick={onStop}
          aria-label="Stop generating"
        >
          <span aria-hidden="true" />
        </button>
      ) : (
        <button
          className="composer-action send-action"
          type="submit"
          disabled={!value.trim()}
          aria-label="Send message"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 19V5m0 0-6 6m6-6 6 6" />
          </svg>
        </button>
      )}
    </form>
  );
}
