import { useState, useEffect, useRef } from "react"
import { useLocation, useNavigate } from "react-router-dom"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import Navbar from "../components/Navbar"

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

interface Source {
  video_id: string
  title: string
  chunk_index: number
  text: string
  start_time: number
}

interface Message {
  role: "user" | "assistant"
  content: string
  sources?: Source[]
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

export default function Chat() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const video: Video | null = state?.video ?? null

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  const bottomRef = useRef<HTMLDivElement>(null)

  const suggestedQuestions = parseQuestions(video?.suggested_questions)

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
    setMessages(prev => [...prev, { role: "assistant", content: "" }])

    try {
      const history = messages
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content }))

      const token = localStorage.getItem("token")
      const response = await fetch(`${BASE_URL}/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          question,
          video_id: video?.youtube_video_id ?? null,
          history,
        }),
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || "Request failed")
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const data = JSON.parse(line.slice(6))
          if (data.token) {
            setMessages(prev => {
              const updated = [...prev]
              const last = updated[updated.length - 1]
              updated[updated.length - 1] = { ...last, content: last.content + data.token }
              return updated
            })
          } else if (data.done) {
            setMessages(prev => {
              const updated = [...prev]
              const last = updated[updated.length - 1]
              updated[updated.length - 1] = { ...last, sources: data.sources }
              return updated
            })
          }
        }
      }
    } catch (err: any) {
      setMessages(prev => prev.slice(0, -1))
      setError(err.message || "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <Navbar />

      {/* Sub-header */}
      <div className="bg-white border-b border-gray-100 px-6 py-3 flex items-center gap-4">
        <button
          onClick={() => navigate("/library")}
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
                    ) : (
                      <div className="[&_ul]:list-disc [&_ul]:pl-4 [&_ul]:space-y-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:space-y-1 [&_strong]:font-semibold [&_p]:mb-2 [&_p:last-child]:mb-0 [&_li]:mb-0.5 [&_code]:font-mono [&_code]:bg-gray-100 [&_code]:px-1 [&_code]:rounded">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      </div>
                    )}
                  </div>

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
                        <div className="space-y-2 mt-2">
                          {msg.sources.map((src, j) => (
                            <div key={j} className="bg-white border border-gray-100 rounded-xl px-4 py-3 text-xs text-gray-600 shadow-sm">
                              <div className="flex items-start justify-between gap-2 mb-1.5">
                                <p className="font-medium text-gray-700 leading-snug">{src.title}</p>
                                {src.start_time != null && (
                                  <a
                                    href={`https://www.youtube.com/watch?v=${src.video_id}&t=${Math.floor(src.start_time)}s`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    onClick={e => e.stopPropagation()}
                                    className="shrink-0 flex items-center gap-1 text-indigo-500 hover:text-indigo-700 transition"
                                  >
                                    <svg viewBox="0 0 24 24" fill="currentColor" className="w-3 h-3">
                                      <path d="M8 5v14l11-7z" />
                                    </svg>
                                    {formatTime(src.start_time)}
                                  </a>
                                )}
                              </div>
                              <p className="line-clamp-3 leading-relaxed text-gray-500">{src.text}</p>
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
            placeholder={video ? "Ask about this video..." : "Ask across your library..."}
            className="flex-1 border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition bg-slate-50"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white px-4 py-2.5 rounded-xl transition flex items-center gap-1.5 text-sm font-medium"
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
