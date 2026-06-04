import { useState, useEffect, useRef } from "react"
import { useLocation, useNavigate } from "react-router-dom"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import Navbar from "../components/Navbar"

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

interface Source {
  video_id?: string
  title: string
  chunk_index?: number
  text: string
  start_time?: number
  url?: string
  source?: "video" | "web"
}

interface TraceStep {
  tool: "search_videos" | "search_web"
  query: string
  count?: number
}

interface Message {
  role: "user" | "assistant"
  content: string
  sources?: Source[]
  trace?: TraceStep[]
}

interface Video {
  id: number
  youtube_video_id: string
  title: string
  channel_name: string
  thumbnail_url: string
  url: string
  summary?: string | null
  suggested_questions?: string | null
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

function parseQuestions(raw?: string | null): string[] {
  if (!raw) return []
  try { return JSON.parse(raw) } catch { return [] }
}

function ReasoningTrace({ trace }: { trace: TraceStep[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-1">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
          className={`w-3 h-3 transition-transform ${open ? "rotate-90" : ""}`}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        </svg>
        {open ? "Hide reasoning" : "Show reasoning"}
      </button>
      {open && (
        <div className="mt-1.5 flex flex-col gap-1">
          {trace.map((step, i) => (
            <div key={i} className="flex items-baseline gap-1.5 text-xs text-gray-500">
              <span className={`shrink-0 font-medium ${step.tool === "search_videos" ? "text-indigo-500" : "text-emerald-500"}`}>
                {step.tool === "search_videos" ? "▶ video" : "⌕ web"}
              </span>
              <span className="italic truncate">&ldquo;{step.query}&rdquo;</span>
              {step.count !== undefined && (
                <span className="shrink-0 text-gray-400">· {step.count}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Chat() {
  const { state } = useLocation()
  const navigate = useNavigate()

  // Persist video in localStorage so it survives refresh
  const videoFromState: Video | null = state?.video ?? null
  const storageKey = videoFromState
    ? `chat_video_${videoFromState.youtube_video_id}`
    : "chat_video_library"
  const messagesKey = videoFromState
    ? `chat_messages_${videoFromState.youtube_video_id}`
    : "chat_messages_library"

  if (videoFromState) {
    localStorage.setItem(storageKey, JSON.stringify(videoFromState))
  }

  const video: Video | null = videoFromState ?? (() => {
    try { return JSON.parse(localStorage.getItem(storageKey) || "null") } catch { return null }
  })()

  const [messages, setMessages] = useState<Message[]>(() => {
    try { return JSON.parse(localStorage.getItem(messagesKey) || "[]") } catch { return [] }
  })
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  const bottomRef = useRef<HTMLDivElement>(null)

  const suggestedQuestions = parseQuestions(video?.suggested_questions)

  // Persist messages on every change
  useEffect(() => {
    localStorage.setItem(messagesKey, JSON.stringify(messages))
  }, [messages, messagesKey])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  function toggleSources(idx: number) {
    setExpandedSources(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || loading) return
    submitQuestion(input)
  }

  async function submitQuestion(question: string) {
    setMessages(prev => [...prev, { role: "user", content: question }])
    setInput("")
    setLoading(true)
    setError("")
    setMessages(prev => [...prev, { role: "assistant", content: "", trace: [] }])

    try {
      const token = localStorage.getItem("token")
      await streamAgent(question, token)
    } catch (err: any) {
      setMessages(prev => prev.slice(0, -1))
      setError(err.message || "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  async function streamAgent(question: string, token: string | null) {
    const response = await fetch(`${BASE_URL}/query/agent`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        question,
        video_id: video?.youtube_video_id ?? null,
        history: messages.slice(-4).map(m => ({ role: m.role, content: m.content })),
      }),
    })

    if (!response.ok) {
      const err = await response.json()
      throw new Error(err.detail || "Request failed")
    }

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ""
    // track pending tool call to pair with its result
    let pendingTool: { tool: string; query: string } | null = null

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split("\n")
      buffer = lines.pop() ?? ""

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue
        const data = JSON.parse(line.slice(6))

        if (data.type === "tool_call") {
          pendingTool = { tool: data.tool, query: data.query }
          setMessages(prev => {
            const updated = [...prev]
            const last = { ...updated[updated.length - 1] }
            last.trace = [...(last.trace ?? []), { tool: data.tool, query: data.query }]
            updated[updated.length - 1] = last
            return updated
          })
        } else if (data.type === "tool_result") {
          pendingTool = null
          setMessages(prev => {
            const updated = [...prev]
            const last = { ...updated[updated.length - 1] }
            const trace = [...(last.trace ?? [])]
            // update the last trace step with the count
            if (trace.length > 0) {
              trace[trace.length - 1] = { ...trace[trace.length - 1], count: data.count }
            }
            last.trace = trace
            updated[updated.length - 1] = last
            return updated
          })
        } else if (data.type === "token") {
          setMessages(prev => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            updated[updated.length - 1] = { ...last, content: last.content + data.token }
            return updated
          })
        } else if (data.type === "done") {
          setMessages(prev => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            updated[updated.length - 1] = { ...last, sources: data.sources }
            return updated
          })
        }
      }
    }

    // suppress unused warning
    void pendingTool
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <Navbar />

      {/* Sub-header */}
      <div className="bg-white border-b border-gray-100 px-6 py-3 flex items-center gap-4">
        <button
          onClick={() => { localStorage.removeItem(messagesKey); navigate("/library") }}
          className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-700 transition"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
          </svg>
          Library
        </button>

        <div className="w-px h-5 bg-gray-200" />

        {video ? (
          <div className="flex items-center gap-3 min-w-0">
            <img src={video.thumbnail_url} alt={video.title} className="w-12 rounded-lg object-cover shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-semibold text-gray-900 truncate">{video.title}</p>
              <p className="text-xs text-gray-400">{video.channel_name}</p>
            </div>
          </div>
        ) : (
          <div>
            <p className="text-sm font-semibold text-gray-900">Entire library</p>
            <p className="text-xs text-gray-400">Answers will cite which video they came from</p>
          </div>
        )}

      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 py-8 space-y-5">
          {messages.length === 0 && (
            <div className="space-y-6 py-8">
              <div className="text-center">
                <p className="text-gray-700 font-medium">
                  {video ? `Ask anything about this video` : "Ask anything across your library"}
                </p>
                {video?.summary && (
                  <p className="text-sm text-gray-500 max-w-md mx-auto mt-2 leading-relaxed">{video.summary}</p>
                )}
              </div>

              {suggestedQuestions.length > 0 && (
                <div className="flex flex-wrap justify-center gap-2">
                  {suggestedQuestions.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => submitQuestion(q)}
                      className="bg-white hover:bg-indigo-50 border border-gray-200 hover:border-indigo-300 text-gray-700 hover:text-indigo-700 rounded-xl px-4 py-2 text-sm transition shadow-sm"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              {msg.role === "user" ? (
                <div className="bg-indigo-600 text-white text-sm px-4 py-3 rounded-2xl rounded-tr-sm max-w-sm shadow-sm">
                  {msg.content}
                </div>
              ) : (
                <div className="flex flex-col gap-2 max-w-xl w-full">
                  <div className="bg-white border border-gray-100 text-gray-800 text-sm px-4 py-3.5 rounded-2xl rounded-tl-sm shadow-sm">
                    {msg.content === "" && loading && i === messages.length - 1 ? (
                      <span className="inline-flex gap-1 items-center">
                        <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                        <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                        <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
                      </span>
                    ) : msg.content ? (
                      <div className="[&_ul]:list-disc [&_ul]:pl-4 [&_ul]:space-y-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:space-y-1 [&_strong]:font-semibold [&_p]:mb-2 [&_p:last-child]:mb-0 [&_li]:mb-0.5 [&_code]:font-mono [&_code]:bg-gray-100 [&_code]:px-1 [&_code]:rounded">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-gray-400 italic text-sm">No answer found.</p>
                    )}
                  </div>

                  {/* Reasoning trace (agent mode) */}
                  {msg.trace && msg.trace.length > 0 && (
                    <ReasoningTrace trace={msg.trace} />
                  )}

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div>
                      <button
                        onClick={() => toggleSources(i)}
                        className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition"
                      >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className={`w-3 h-3 transition-transform ${expandedSources.has(i) ? "rotate-90" : ""}`}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                        </svg>
                        {expandedSources.has(i) ? "Hide sources" : `${msg.sources.length} source${msg.sources.length !== 1 ? "s" : ""}`}
                      </button>

                      {expandedSources.has(i) && (
                        <div className="mt-1.5 flex flex-col gap-2">
                          {msg.sources.slice(0, 4).map((src, j) => (
                            <div key={j} className="flex gap-2 text-xs text-gray-500">
                              <span className={`shrink-0 mt-0.5 ${src.source === "web" ? "text-emerald-500" : "text-indigo-400"}`}>
                                {src.source === "web" ? "⌕" : "▶"}
                              </span>
                              <div className="min-w-0">
                                {src.source === "web" ? (
                                  <a href={src.url} target="_blank" rel="noopener noreferrer"
                                    className="font-medium text-gray-600 hover:text-emerald-600 hover:underline transition">
                                    {src.title}
                                  </a>
                                ) : (
                                  <a
                                    href={`https://www.youtube.com/watch?v=${src.video_id}&t=${Math.floor(src.start_time ?? 0)}s`}
                                    target="_blank" rel="noopener noreferrer"
                                    className="font-medium text-gray-600 hover:text-indigo-600 hover:underline transition">
                                    {video ? formatTime(src.start_time ?? 0) : `${src.title} · ${formatTime(src.start_time ?? 0)}`}
                                  </a>
                                )}
                                <p className="text-gray-400 line-clamp-1 mt-0.5">{src.text}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {error && (
            <div className="flex justify-center">
              <p className="text-red-500 text-sm bg-red-50 border border-red-100 rounded-xl px-4 py-2">{error}</p>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-100 px-4 py-4">
        <form onSubmit={handleAsk} className="max-w-2xl mx-auto flex gap-2.5">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={video ? "Ask about this video" : "Ask across your library"}
            className="flex-1 border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition bg-slate-50"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="disabled:opacity-50 text-white px-4 py-2.5 rounded-xl transition flex items-center gap-1.5 text-sm font-medium bg-indigo-600 hover:bg-indigo-700"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
            Ask
          </button>
        </form>
      </div>
    </div>
  )
}
